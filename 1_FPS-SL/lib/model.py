import torch
import torch.nn as nn
from random import sample
import numpy as np
import torch.nn.functional as F
from torch.nn.modules import loss
from lib.pvtv2_mil import pvt_v2_b2, SupConPVT, SupConPVT_CL
from utils.utils_loss_multilabel import jaccard_distance, get_confident_indices
from utils.stable_queue import StableDynamicQueue
import torch.distributed as dist
from lib.hagmil import IAMBlock, AggregationBlockv2
import time


def create_pvt_encoder(args, pretrained=False, only_CL=False):
    if args.cls_task == 'multi_cls':
        if only_CL:
            model = SupConPVT_CL(head='mlp', feat_dim=args.low_dim, num_class=args.num_class)
        else:
            model = SupConPVT(head='mlp', feat_dim=args.low_dim, num_class=args.num_class)
    elif args.cls_task == 'multi_label':
        if only_CL:
            model = SupConPVT_CL(head='mlp', feat_dim=args.low_dim, num_class=args.num_class)
        else:
            model = SupConPVT(args=args, head='mlp', feat_dim=args.low_dim, num_class=args.num_class)

            # model = SupConPVT(head='mlp', feat_dim=args.low_dim, num_class=args.num_class)
            # model = SupConPVT_MultiLabel(head='mlp', feat_dim=args.low_dim, num_class=args.num_class) # multi CL space
    # elif args.cls_task == 'multi_task':
        # model = SupConPVT_addSeg(head='mlp', feat_dim=args.low_dim, num_class=args.num_class)
    else:
        raise NotImplementedError

    if pretrained:
        # path = '/mnt/data/yizhenyu/data/HP识别/workspace/PiCO/pretrained/PolypPVT.pth'
        # path = '/mnt/data/yizhenyu/data/HP识别/workspace/PiCO/pretrained/pvt_v2_b2.pth'
        path = args.pretrained_path
        print('load pretrained backbone from:', path)

        # save_model = torch.load(path)
        # model_dict = model.state_dict()
        # state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        # model_dict.update(state_dict)
        # model.load_state_dict(model_dict)

        # # 打印加载结果
        # print("===== 成功加载参数 =====")
        # for k in state_dict:
        #     print(f"load: {k} ")
        # print("=====  =====")

        save_model = torch.load(path)  # 加载预训练模型
        model_dict = model.state_dict()  # 获取当前模型参数字典
        # print(model_dict.keys(), save_model.keys())
        # 构建参数映射字典
        state_dict = {}
        if 'PolypPVT' in path:
            for key in model_dict:
                # 处理encoder部分：model.encoder <-> save_model.backbone
                if key.startswith("encoder."):
                    src_key = key.replace("encoder.", "backbone.", 1)
                    if src_key in save_model:
                        state_dict[key] = save_model[src_key]
                
                # 处理seg_head部分：model.seg_head <-> save_model
                elif key.startswith("seg_head.") and args.add_seg_head:
                    src_key = key.replace("seg_head.", "", 1)  # 直接映射顶层键
                    if src_key in save_model:
                        state_dict[key] = save_model[src_key]

        elif 'pvt_v2_b2' in path and 'MIL-HP_yzy' not in path:
            for key in model_dict:
                # 处理encoder部分：model.encoder <-> save_model.backbone
                if key.startswith("encoder."):
                    src_key = key.replace("encoder.", "", 1)
                    if src_key in save_model:
                        state_dict[key] = save_model[src_key]

        elif 'MIL-HP_yzy' in path:
            for key in model_dict:
                # 处理encoder部分：model.encoder <-> save_model.backbone
                if key.startswith("encoder."):
                    src_key = key.replace("encoder.", "feature_extractor.", 1)
                    if src_key in save_model:
                        state_dict[key] = save_model[src_key]
                elif key.startswith("fc_bag."):
                    src_key = key.replace("fc_bag.", "feature_extractor.head.", 1)
                    if src_key in save_model:
                        state_dict[key] = save_model[src_key]

        # 更新并加载参数
        model_dict.update(state_dict)
        model.load_state_dict(model_dict, strict=False)  # 允许部分加载

        # # 打印加载结果
        # print("===== load params =====")
        # for k in state_dict:
        #     print(f"{k}")
        # print("\n=====  =====")

    return model

class Mean(nn.Module):
    def __init__(self, dim=1, keepdim=False):
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x):
        return x.mean(dim=self.dim, keepdim=self.keepdim)


class Bag_aggregate_head_only_fc(nn.Module):
    def __init__(self, out_dim=2, feature_dim=512, hidden_dim=512):
        super().__init__()

        self.fc = nn.Linear(feature_dim, out_dim)

class Bag_aggregate_head(nn.Module):
    def __init__(self, out_dim=2, feature_dim=512, hidden_dim=512, args=None):
        super().__init__()

        if args.aggregate == 'v1':
            self.aggregate = IAMBlock(in_dim=feature_dim, out_dim=hidden_dim, final_dim=hidden_dim, size=[feature_dim,512,256], dropout=0.25, gate=True)
        elif args.aggregate == 'v2':
            self.aggregate = AggregationBlockv2(in_dim=feature_dim, mid_dim=hidden_dim, final_dim=hidden_dim, size=[feature_dim,512,256], dropout=0.25, gate=True)
        self.aggregate_type = args.aggregate
        self.fc = nn.Linear(feature_dim, out_dim)
    
    
    def forward(self, x, q):
        # x: [1,K,E]
        # q: [1,K,E']
        # score: [1,K,C]
        # time_1 = time.time()
        if self.aggregate_type == 'v1':
            x, A, A_raw = self.aggregate(x)
        elif self.aggregate_type == 'v2':
            x, A, A_raw = self.aggregate(x, q) #[1,E]
        # x = x.mean(dim=1, keepdim=False)

        # time_agg = time.time() 
        # print('2-time_agg:', time_agg - time_1)
        # print('2-time_fc:', time.time() - time_agg)

        return self.fc(x), A_raw

class PiCO(nn.Module):

    def __init__(self, args, base_encoder):
        super().__init__()
        
        pretrained = args.pretrained
        self.cls_task = args.cls_task
        self.num_class = args.num_class

        # self.threshold = threshold
        self.thre_vec_cls=torch.tensor([0.5]*8)
        self.thre_vec_prot=torch.tensor([0.5]*8)
        ## judge self.threshold is list or not, if is list, transform to tensor; if is float, repeat and transform to tensor
        # if isinstance(self.threshold, list):
        #     self.threshold = torch.tensor(self.threshold)
        # else:
        #     self.threshold = torch.tensor([threshold]*args.num_class)

        self.stable_queue = args.stable_queue
        self.conf_thres = args.conf_thres
        self.instance_batch_size = 40

        # resnet encoder
        if args.arch == 'resnet18':
            self.encoder_q = base_encoder(num_class=args.num_class, feat_dim=args.low_dim, name=args.arch, pretrained=pretrained)
            # momentum encoder
            self.encoder_k = base_encoder(num_class=args.num_class, feat_dim=args.low_dim, name=args.arch, pretrained=False)
        # pvt encoder
        elif args.arch == 'pvt_v2_b2':
            self.encoder_q = create_pvt_encoder(args, pretrained=pretrained)
            self.encoder_k = create_pvt_encoder(args, pretrained=False)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        if self.cls_task == 'multi_cls':
            # create the queue
            self.register_buffer("queue", torch.randn(args.moco_queue, args.low_dim)) # L, D
            self.register_buffer("queue_pseudo", torch.zeros(args.moco_queue)) # L
            self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))     

            self.register_buffer("prototypes", torch.zeros(args.num_class,args.low_dim)) # 2, D
        elif self.cls_task == 'multi_label':
            # create the queue
            self.register_buffer("queue", torch.randn(args.moco_queue, args.low_dim)) # L, D
            self.register_buffer("queue_pseudo", torch.zeros(args.moco_queue, self.num_class)) # L, C
            self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))  # 1

            self.register_buffer("prototypes", torch.zeros(self.num_class+1, args.low_dim)) # C+1, D
            
            if self.stable_queue:
                print('apply stable queue')
                # self.queue_entry_batch = -torch.ones(args.moco_queue,)
                # self.register_buffer("queue_entry_batch", -torch.ones(args.moco_queue,)) # C+1, D
                self.queue_assis = StableDynamicQueue(queue_size=args.moco_queue, buffer_size=args.batch_size, num_classes=self.num_class)

        else:
            raise NotImplementedError

        self.queue = F.normalize(self.queue, dim=0)

        # create bag predict head
        self.add_bag_head = args.add_bag_head
        self.pseudo_bag_sup = args.pseudo_bag_sup
        if self.add_bag_head:
            # self.bag_predict_head = nn.Linear(512, args.num_class)
            self.bag_predict_head = Bag_aggregate_head(out_dim=2, feature_dim=512, hidden_dim=512, args=args)
        # elif self.pseudo_bag_sup:
        #     self.bag_predict_head = Bag_aggregate_head_only_fc(out_dim=2, feature_dim=512, hidden_dim=512)

        # self.register_buffer('dummy', torch.zeros(1)) # for bag_predict_head


    def forward(self, img_q, im_k=None, partial_Y=None, args=None, eval_only=False, start_upd_prot=False, semi_stage2=False, task='img', eval_all=False):

        if eval_all:
            logits_clsHead, q, feat_encoder, logits_clsHead_bag = self.encoder_q(img_q, task='all')
            score_clsHead = torch.sigmoid(logits_clsHead)
            # compute protoypical logits
            prototypes = self.prototypes.clone().detach() #[C+1,D]
            prototype_normal = prototypes[0].unsqueeze(0) # [1,D]
            prototypes_abnormal = prototypes[1:] # [C,D]

            ## compare between abnormal and normal prototype
            if args.proto_scoring == 'v1':
                prototype_normal_expanded = prototype_normal.expand(self.num_class, -1)  # [num_class, D]
                prototypes_abnormal_expanded = prototypes_abnormal.unsqueeze(1)  # [num_class, 1, D]
                prototype_all = torch.cat([prototype_normal_expanded.unsqueeze(1), prototypes_abnormal_expanded], dim=1)  # [num_class, 2, D]
                logits_prot = torch.einsum('b d, c n d -> b c n',
                    q,  # [B, D]
                    prototype_all  # [num_class, 2, D]
                )  # [B, num_class: 8, 2]
                score_prot = torch.softmax(logits_prot, dim=2)[:, :, 1]  # [B, num_class]   

            ## normalize the sims of abnormal
            elif args.proto_scoring == 'v2':
                score_prot = torch.exp(q @ prototypes_abnormal.t()/args.temperature)
                score_prot = (score_prot - score_prot.min()) / (score_prot.max() - score_prot.min())

            logits_bag, A_raw = self.bag_predict_head(feat_encoder.unsqueeze(0), q)

            return score_clsHead, q, feat_encoder, score_prot, logits_bag

        # ------------------------------- #
        if eval_only and task == 'img':
            # for testing, no grad compute map
            logits_clsHead, q, feat_encoder, logits_clsHead_bag = self.encoder_q(img_q, task)
            score_clsHead = torch.sigmoid(logits_clsHead)
            # compute protoypical logits
            prototypes = self.prototypes.clone().detach() #[C+1,D]
            prototype_normal = prototypes[0].unsqueeze(0) # [1,D]
            prototypes_abnormal = prototypes[1:] # [C,D]

            # score_prot = torch.zeros((q.shape[0], self.num_class)).to(q.device) #[B,C]
            # for i in range(self.num_class):
            #     prototype_i = torch.cat([prototype_normal, prototypes_abnormal[i].unsqueeze(0)], dim=0)
            #     logits_prot_i = torch.mm(q, prototype_i.t()) #[B,2]
            #     score_prot_i = torch.softmax(logits_prot_i, dim=1)[:,1]
            #     score_prot[:,i] = score_prot_i
            
            # - q: [B, D]
            # - prototype_normal: [D]
            # - prototypes_abnormal: [num_class, D]
            ## compare between abnormal and normal prototype
            if args.proto_scoring == 'v1':
                prototype_normal_expanded = prototype_normal.expand(self.num_class, -1)  # [num_class, D]
                prototypes_abnormal_expanded = prototypes_abnormal.unsqueeze(1)  # [num_class, 1, D]
                prototype_all = torch.cat([prototype_normal_expanded.unsqueeze(1), prototypes_abnormal_expanded], dim=1)  # [num_class, 2, D]
                logits_prot = torch.einsum('b d, c n d -> b c n',
                    q,  # [B, D]
                    prototype_all  # [num_class, 2, D]
                )  # [B, num_class, 2]
                score_prot = torch.softmax(logits_prot, dim=2)[:, :, 1]  # [B, num_class]   

            ## normalize the sims of abnormal
            elif args.proto_scoring == 'v2':
                score_prot = torch.exp(q @ prototypes_abnormal.t()/args.temperature)
                score_prot = (score_prot - score_prot.min()) / (score_prot.max() - score_prot.min())

            return score_clsHead, q, feat_encoder, score_prot

        # ------------------------------- #
        if task == 'bag' and self.add_bag_head:
            # time_s = time.time()
            logits_clsHead, q, feat_encoder, logits_clsHead_bag = self.encoder_q(img_q, task='img')
            score_clsHead = torch.sigmoid(logits_clsHead)

            # z_norm = q.squeeze()
            # cos_sim = z_norm()@z_norm.T  # [-1, 1]
            # squared_distance = 2 * (1 - cos_sim)  
            # combined_adj_sim = np.exp(-1.0 * squared_distance / (2 * 1 ** 2)) # [K,K]

            # # mask = (combined_adj_sim >= 0.7).float()  # shape: [n, n]
            # mask = torch.sigmoid(50*(combined_adj_sim-0.7))

            # row_sums = torch.sum(combined_adj_sim * mask, dim=1, keepdim=True)  # shape: [n, 1]

            # inv_row_sums = 1.0 / (row_sums + 1e-8) # shape: [n, 1]
            # mask_drop = mask * inv_row_sums + (1 - mask) * 1.0  # shape: [n, n]

            logits, A_raw = self.bag_predict_head(feat_encoder.unsqueeze(0), q)
            # print('1-bag predict time:', time.time() - time_en)
            # features = features.mean(dim=0,keepdim=True) # mean pooling: 1*D
            return logits, A_raw, score_clsHead # 1*C, 1
            
        # ------------------------------- #
        if task == 'seg' and args.add_seg_head:
            P1, P2 = self.encoder_q(img_q, task)
            return P1, P2


        # ------------------------------- #
        # img train #
    
        '''1) calculate CLS-1 from cls head, and sametime combine the maybe [true label]'''
        # time1 = time.time()
        output, q, feat_encoder, output_bag = self.encoder_q(img_q, task)
        # print('1-encode time:', time.time() - time1)
        # predicted_scores = torch.sigmoid(output) * partial_Y ## here, emplicitly inject true labels ##
        # # max_scores, pseudo_labels_b = torch.max(predicted_scores, dim=1)
        # pseudo_labels_b = (predicted_scores > self.threshold).float() # [B,C]
        # predicted_scores = (torch.sigmoid(output) > self.threshold).float()
        # pseudo_labels_b = ((predicted_scores + partial_Y - 1) > 0).float() 

        mask = (partial_Y != 0.5).float() # [B,C], {0,1,0.5}
        # print('percent of partial labels:', mask.sum() / (8 * mask.shape[0]))
        predicted_scores = partial_Y * mask + torch.sigmoid(output) * (1 - mask)
        pseudo_labels_b = (predicted_scores > self.thre_vec_cls.to(predicted_scores.device)).long()
        
        # print('0', output)
        # print('1', partial_Y)
        # print('2', pseudo_labels_b)
        # using partial labels to filter out negative labels

        '''2) calculate CLS-2 from prototypes'''
        # compute protoypical logits
        prototypes = self.prototypes.clone().detach() #[C+1,D]
        prototype_normal = prototypes[0].unsqueeze(0) # [1,D]
        prototypes_abnormal = prototypes[1:] # [C,D]
        # score_prot = torch.zeros((q.shape[0], self.num_class)).to(q.device) #[B,C]
        # for i in range(self.num_class):
        #     prototype_i = torch.cat([prototype_normal, prototypes_abnormal[i].unsqueeze(0)], dim=0)
        #     logits_prot_i = torch.mm(q, prototype_i.t()) #[B,2]
        #     score_prot_i = torch.softmax(logits_prot_i, dim=1)[:,1]
        #     score_prot[:,i] = score_prot_i
        
        # - q: [B, D]
        # - prototype_normal: [D]
        # - prototypes_abnormal: [num_class, D]
        ## compare between abnormal and normal prototype
        if args.proto_scoring == 'v1':
            prototype_normal_expanded = prototype_normal.expand(self.num_class, -1)  # [num_class, D]
            prototypes_abnormal_expanded = prototypes_abnormal.unsqueeze(1)  # [num_class, 1, D]
            prototype_all = torch.cat([prototype_normal_expanded.unsqueeze(1), prototypes_abnormal_expanded], dim=1)  # [num_class, 2, D]
            logits_prot = torch.einsum('b d, c n d -> b c n',
                q,  # [B, D]
                prototype_all  # [num_class, 2, D]
            )  # [B, num_class, 2]
            score_prot = torch.softmax(logits_prot, dim=2)[:, :, 1]  # [B, num_class]   

        ## normalize the sims of abnormal
        elif args.proto_scoring == 'v2':
            score_prot = torch.exp(q @ prototypes_abnormal.t()/args.temperature)
            score_prot = (score_prot - score_prot.min()) / (score_prot.max() - score_prot.min())
        score_prot_b = (score_prot > self.thre_vec_prot.to(score_prot.device)).long()
        # mask_valid = (score_prot > args.pos_conf_thres / 0.5 * self.thre_vec_prot) | (score_prot < args.neg_conf_thres / 0.5 * (1-self.thre_vec_prot)) 

        # compute key features 
        with torch.no_grad():  # no gradient 
            self._momentum_update_key_encoder(args)  # update the momentum encoder
            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)
            _, k, _, logits_k_bag = self.encoder_k(im_k, task)
            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)
            logits_k_bag = self._batch_unshuffle_ddp(logits_k_bag, idx_unshuffle)
            # print(logits_k_bag)


        '''3) Update prototypes with CLS-1 (and maybe [true label])'''
        # update momentum prototypes with pseudo labels
        # q: [B,D]; pseudo_labels_b: [B,C]; self.prototypes: [C+1,D]

        # dist.barrier()
        if not semi_stage2:
            q_concat = concat_all_gather(q)
            pseudo_labels_b_concat = concat_all_gather(pseudo_labels_b)
            self._update_prototypes(q_concat, pseudo_labels_b_concat, args)
        elif start_upd_prot: # when semi_stage2 and start_upd_prot:
            q_concat = concat_all_gather(q)
            pseudo_labels_b_concat = concat_all_gather(pseudo_labels_b)

            indices_confident_class = get_confident_indices(predicted_scores, self.thre_vec_cls.to(predicted_scores.device), mask_labeled=mask, pos_conf_per=args.pos_conf_percent, neg_conf_per=args.neg_conf_percent) #[B,C]
            indices_confident_sample = (indices_confident_class.sum(1) == indices_confident_class.shape[1]) #[B]
            indices_confident_class_concat = concat_all_gather(indices_confident_class) #[B,C]

            self._update_prototypes(q_concat, pseudo_labels_b_concat, args, mask_conf=indices_confident_class_concat)

            # filter out unconfident labels when using SupCL
            # k = k[indices_confident_sample]
            q, pseudo_labels_b, k = q[indices_confident_sample], pseudo_labels_b[indices_confident_sample], k[indices_confident_sample]

        '''filter out unconfident labels when using SupCL'''
        if start_upd_prot and semi_stage2:
            mask_valid_prot = get_confident_indices(score_prot, self.thre_vec_cls.to(predicted_scores.device), mask_labeled=mask, pos_conf_per=args.pos_conf_percent, neg_conf_per=args.neg_conf_percent) #[B,C]
        else:   
            mask_valid_prot = None

        '''4) Update queue with CLS-1 (and maybe [true label])'''
        features = torch.cat((q, k, self.queue.clone().detach()), dim=0)
        pseudo_labels = torch.cat((pseudo_labels_b, pseudo_labels_b, self.queue_pseudo.clone().detach()), dim=0)
        # to calculate SupCon Loss using pseudo_labels
        
        # dequeue and enqueue
        self._dequeue_and_enqueue(k, pseudo_labels_b, args, start_upd_prot)

        # '''5) casual one-bag prediction'''''
        # logits_one_bag = self.encoder_q.fc_bag(feat_encoder) if self.pseudo_bag_sup else None

        return output, features, pseudo_labels, score_prot_b, output_bag, mask_valid_prot, logits_k_bag

    def switch_learnabel_weights(self, stage='1'):
        if stage == '1':
            for param in self.encoder_q.parameters():
                param.requires_grad = True  
            for param in self.bag_predict_head.parameters():
                param.requires_grad = False
            print('switch to stage 1')
        
        elif stage == '2':
            for param in self.encoder_q.parameters():
                param.requires_grad = False  
            for param in self.bag_predict_head.parameters():
                param.requires_grad = True     
            print('switch to stage 2')

    def reset_queue(self):
        self.queue_ptr[0] = 0 #torch.zeros(1, dtype=torch.long)

    def set_thre_vec(self, thre_vec_cls, thre_vec_prot):
        self.thre_vec_cls = thre_vec_cls
        self.thre_vec_prot = thre_vec_prot

    @torch.no_grad()
    def _momentum_update_key_encoder(self, args):
        """
        update momentum encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * args.moco_m + param_q.data * (1. - args.moco_m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, labels, args, start_upd_prot):
        # gather keys before updating queue
        keys = concat_all_gather(keys) #[B,D]
        labels = concat_all_gather(labels) #[B,C]

        batch_size = keys.shape[0]
        if self.stable_queue and start_upd_prot:
            # print('2, start_upd_prot:', start_upd_prot)
            self.queue_ptr, self.queue, self.queue_pseudo = \
            self.queue_assis.enqueue_batch(self.queue_ptr, self.queue, self.queue_pseudo, 
                                    keys, labels)

        else:
            ptr = int(self.queue_ptr) # [1]
            # print('val:',args.moco_queue, batch_size)
            assert args.moco_queue % batch_size == 0  # for simplicity

            # replace the keys at ptr (dequeue and enqueue)
            self.queue[ptr:ptr + batch_size] = keys
            self.queue_pseudo[ptr:ptr + batch_size] = labels
            ptr = (ptr+ batch_size) % args.moco_queue  # move pointer

            self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)

        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def efficient_chunk_processing(self, img_q, chunk_size=64):
        max_instances = img_q.size(0)
        features = []
        
        # 使用CUDA流并行处理
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            for i in range(0, max_instances, chunk_size):
                # 分块处理并保留梯度
                chunk = img_q[i:i+chunk_size].contiguous()
                
                # 使用混合精度加速
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    chunk_features = self.encoder_q(chunk, only_feat=True)
                
                # 非阻塞传输
                features.append(chunk_features.to(device='cuda', non_blocking=True))
                
                # 显存优化
                if i % (chunk_size*2) == 0:
                    torch.cuda.empty_cache()
    
        # 异步拼接
        with torch.cuda.stream(torch.cuda.default_stream()):
            return torch.cat(features, dim=0).contiguous()


    def _update_prototypes(self, q, pseudo_labels_b, args, mask_conf=None):
        """
        原型更新函数
        :param q: 当前批次特征 [B, D]
        :param pseudo_labels_b: 伪标签矩阵 [B, C], {0,1}
        :param args.proto_m: 动量系数 (0.9)
        :param mask_conf: [B,C] or None, confident mask, {0,1}
        """
        # --------------------------------------------------
        # 参数检查
        # --------------------------------------------------
        # print(q.shape, pseudo_labels_b.shape)
        # print(pseudo_labels_b[:2])
        assert q.dim() == 2, "特征q必须是二维张量 [B, D]"
        assert pseudo_labels_b.dim() == 2, "伪标签必须是二维张量 [B, C]"
        B, C = pseudo_labels_b.shape
        device = q.device
        
        # --------------------------------------------------
        # 更新背景原型（索引0）
        # --------------------------------------------------
        # 筛选无标签样本 (全0行)
        bg_mask = (pseudo_labels_b.sum(dim=1) == 0)
        if mask_conf is not None:
            bg_mask *= (mask_conf.sum(dim=1) == mask_conf.shape[1])  # [B]

        if bg_mask.sum() > 0:
            bg_feats = q[bg_mask]  # [num_bg, D]
            # 动量更新背景原型
            self.prototypes[0] = (
                self.prototypes[0] * args.proto_m + 
                (1 - args.proto_m) * bg_feats.mean(dim=0)
            )

        # --------------------------------------------------
        # 更新类别原型（索引1~C）
        # --------------------------------------------------
        for c in range(C):  # 遍历每个标签类别 (c: 0~C-1对应原型的1~C)
            # 筛选当前类别的正样本
            c_mask = pseudo_labels_b[:, c].bool()  # [B]
            if not c_mask.any():
                continue  # 无正样本则跳过
            
            # --------------------------------------------------
            # 核心：计算加权特征均值
            # --------------------------------------------------
            # 1. 获取正样本特征 [num_pos, D]
            c_feats = q[c_mask]
            
            # 2. 计算每个样本的权重 (1/该样本的标签数) [num_pos]
            with torch.no_grad():
                sample_weights = 1.0 / pseudo_labels_b[c_mask].sum(dim=1)  # 归一化权重
                if mask_conf is not None:
                    sample_weights *= mask_conf[c_mask, c]
                
            # 3. 加权平均 (防止除零)
            weighted_sum = (c_feats * sample_weights.unsqueeze(1)).sum(dim=0)  # [D]
            total_weight = sample_weights.sum()
            
            # 4. 动量更新（处理极小权重情况）
            if total_weight > 1e-6:
                c_proto_idx = c + 1  # 原型索引偏移（因为0是背景）
                new_mean = weighted_sum / total_weight
                self.prototypes[c_proto_idx] = (
                    self.prototypes[c_proto_idx] * args.proto_m + 
                    (1 - args.proto_m) * new_mean
                )

        self.prototypes = F.normalize(self.prototypes, p=2, dim=-1).detach()
        return None

    # def _update_prototypes(self, q, pseudo_labels_b, args, mask_conf=None):
    #     """
    #     原型更新函数（含背景类）
    #     :param q: 当前批次特征 [B, D]
    #     :param pseudo_labels_b: 伪标签矩阵 [B, C], {0,1} (不含背景类)
    #     :param args.proto_m: 动量系数 (0.9)
    #     :param mask_conf: [B,C] or None, confident mask, {0,1}
    #     """
    #     assert q.dim() == 2, "特征q必须是二维张量 [B, D]"
    #     assert pseudo_labels_b.dim() == 2, "伪标签必须是二维张量 [B, C]"
    #     B, C = pseudo_labels_b.shape
    #     device = q.device

    #     # --------------------------------------------------
    #     # 构造伪标签矩阵（含背景类）
    #     # --------------------------------------------------
    #     L = pseudo_labels_b.float()  # [B, C]

    #     # 添加背景类伪标签（当其他类全为0时为1）
    #     bg_labels = (L.sum(dim=1, keepdim=True) == 0).float()  # [B, 1]
    #     if mask_conf is not None: 
    #        bg_labels = bg_labels * (mask_conf.sum(dim=1, keepdim=True) == 0).float()
    #     L = torch.cat([bg_labels, L], dim=1)  # [B, C+1] (背景类在第0列)

    #     # 应用置信度掩码
    #     if mask_conf is not None:
    #         mask_conf = mask_conf.float()
    #         L[:, 1:] = L[:, 1:] * mask_conf  # 只保留高置信度样本

    #     # --------------------------------------------------
    #     # 矩阵运算更新所有类别（包括背景类）
    #     # --------------------------------------------------
    #     L_T_L = torch.matmul(L.t(), L)  # [C+1, C+1]
    #     L_T_q = torch.matmul(L.t(), q)  # [C+1, D]

    #     # 添加正则化项防止矩阵不可逆
    #     identity = torch.eye(C + 1, device=device) * 1e-5
    #     L_T_L_inv = torch.inverse(L_T_L + identity)  # [C+1, C+1]

    #     # 计算新原型 CPt*
    #     CPt_star = torch.matmul(L_T_L_inv, L_T_q)  # [C+1, D]

    #     # 动量更新（含背景类）
    #     if not hasattr(self, 'prototypes'):
    #         self.prototypes = CPt_star.clone()  # 初始化原型
    #     else:
    #         self.prototypes = (
    #             args.proto_m * self.prototypes + 
    #             (1 - args.proto_m) * CPt_star
    #         )

    #     # 归一化
    #     self.prototypes = F.normalize(self.prototypes, p=2, dim=-1).detach()
    #     return None


    def Queue_analyze_distribution(self):
        return self.queue_assis.analyze_distribution(self.queue_pseudo)
    # def Frozen_Unfrozen_params_between_trainingModes(self, bag_train):
    #     if bag_train:
    #         print('Frozen_Unfrozen_params_between_trainingModes: bag_train; The unfrozen parameters:')
    #         for name, param in self.named_parameters():
    #             if 'bag_predict_head' in name:
    #                 param.requires_grad = True
    #                 print(name, end=' ')
    #             if 'encoder_q' in name and 'fc' in name:
    #                 param.requires_grad = False
    #             if 'encoder_q' in name and 'head' in name:
    #                 param.requires_grad = False
            
    #     else:
    #         print('Frozen_Unfrozen_params_between_trainingModes: not bag_train; The frozen parameters:')
    #         for name, param in self.named_parameters():
    #             if 'bag_predict_head' in name:
    #                 param.requires_grad = False
    #                 print(name, end=' ')
    #             if 'encoder_q' in name and 'fc' in name:
    #                 param.requires_grad = True
    #             if 'encoder_q' in name and 'head' in name:
    #                 param.requires_grad = True
              
# class PiCO_CL(nn.Module):

    # def __init__(self, args, base_encoder, threshold=0.5):
    #     super().__init__()
        
    #     pretrained = args.pretrained
    #     self.cls_task = args.cls_task
    #     self.num_class = args.num_class
    #     self.threshold = threshold

    #     # resnet encoder
    #     if args.arch == 'resnet18':
    #         self.encoder_q = base_encoder(num_class=args.num_class, feat_dim=args.low_dim, name=args.arch, pretrained=pretrained)
    #         # momentum encoder
    #         self.encoder_k = base_encoder(num_class=args.num_class, feat_dim=args.low_dim, name=args.arch, pretrained=False)
    #     # pvt encoder
    #     elif args.arch == 'pvt_v2_b2':
    #         self.encoder_q = create_pvt_encoder(args, pretrained=pretrained, only_CL=True)
    #         self.encoder_k = create_pvt_encoder(args, pretrained=False, only_CL=True)

    #     for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
    #         param_k.data.copy_(param_q.data)  # initialize
    #         param_k.requires_grad = False  # not update by gradient



    #     if self.cls_task == 'multi_cls':
    #         # create the queue
    #         self.register_buffer("queue", torch.randn(args.moco_queue, args.low_dim)) # L, D
    #         self.register_buffer("queue_pseudo", torch.zeros(args.moco_queue)) # L
    #         self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))     

    #         self.register_buffer("prototypes", torch.zeros(args.num_class,args.low_dim)) # 2, D
    #     elif self.cls_task == 'multi_label':
    #         # create the queue
    #         self.register_buffer("queue", torch.randn(args.moco_queue, args.low_dim)) # L, D
    #         self.register_buffer("queue_pseudo", torch.zeros(args.moco_queue, self.num_class)) # L, C
    #         self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))  # 1

    #         self.register_buffer("prototypes", torch.zeros(args.num_class+1, args.low_dim)) # C+1, D
    #     else:
    #         raise NotImplementedError
    #     self.queue = F.normalize(self.queue, dim=0)


    # @torch.no_grad()
    # def _momentum_update_key_encoder(self, args):
    #     """
    #     update momentum encoder
    #     """
    #     for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
    #         param_k.data = param_k.data * args.moco_m + param_q.data * (1. - args.moco_m)

    # @torch.no_grad()
    # def _dequeue_and_enqueue(self, keys, labels, args):
    #     # gather keys before updating queue
    #     keys = concat_all_gather(keys) #[B,D]
    #     labels = concat_all_gather(labels) #[B,C]

    #     batch_size = keys.shape[0]

    #     ptr = int(self.queue_ptr) # [1]
    #     # print('val:',args.moco_queue, batch_size)
    #     assert args.moco_queue % batch_size == 0  # for simplicity

    #     # replace the keys at ptr (dequeue and enqueue)
    #     self.queue[ptr:ptr + batch_size] = keys
    #     self.queue_pseudo[ptr:ptr + batch_size] = labels
    #     ptr = (ptr+ batch_size) % args.moco_queue  # move pointer

    #     self.queue_ptr[0] = ptr


    # @torch.no_grad()
    # def _batch_shuffle_ddp(self, x):
    #     """
    #     Batch shuffle, for making use of BatchNorm.
    #     *** Only support DistributedDataParallel (DDP) model. ***
    #     """
    #     # gather from all gpus
    #     batch_size_this = x.shape[0]
    #     x_gather = concat_all_gather(x)
    #     batch_size_all = x_gather.shape[0]

    #     num_gpus = batch_size_all // batch_size_this

    #     # random shuffle index
    #     idx_shuffle = torch.randperm(batch_size_all).cuda()

    #     # broadcast to all gpus
    #     torch.distributed.broadcast(idx_shuffle, src=0)

    #     # index for restoring
    #     idx_unshuffle = torch.argsort(idx_shuffle)

    #     # shuffled index for this gpu
    #     gpu_idx = torch.distributed.get_rank()
    #     idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

    #     return x_gather[idx_this], idx_unshuffle

    # @torch.no_grad()
    # def _batch_unshuffle_ddp(self, x, idx_unshuffle):
    #     """
    #     Undo batch shuffle.
    #     *** Only support DistributedDataParallel (DDP) model. ***
    #     """
    #     # gather from all gpus
    #     batch_size_this = x.shape[0]
    #     x_gather = concat_all_gather(x)

    #     batch_size_all = x_gather.shape[0]

    #     num_gpus = batch_size_all // batch_size_this

    #     # restored index for this gpu
    #     gpu_idx = torch.distributed.get_rank()
    #     idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

    #     return x_gather[idx_this]

    # def forward(self, img_q, im_k=None, partial_Y=None, args=None, eval_only=False, bag_predict=False):

    #     if eval_only:
    #         # for testing, no grad compute map
    #         q, feat_encoder = self.encoder_q(img_q)
    #         # compute protoypical logits
    #         prototypes = self.prototypes.clone().detach() #[C+1,D]
    #         prototype_normal = prototypes[0].unsqueeze(0) # [1,D]
    #         prototypes_abnormal = prototypes[1:] # [C,D]
    #         score_prot = torch.zeros((q.shape[0], self.num_class)).to(q.device) #[B,C]
    #         for i in range(self.num_class):
    #             prototype_i = torch.cat([prototype_normal, prototypes_abnormal[i].unsqueeze(0)], dim=0)
    #             logits_prot_i = torch.mm(q, prototype_i.t()) #[B,2]
    #             score_prot_i = torch.softmax(logits_prot_i, dim=1)[:,1]
    #             score_prot[:,i] = score_prot_i

    #         return q, feat_encoder, score_prot
        
            
    #     # ------------------------------- #
    #     '''calculate CLS-1 from cls head, and sametime combine the maybe [true label]'''
    #     q, feat_dummy = self.encoder_q(img_q)
    #     # predicted_scores = torch.sigmoid(output) * partial_Y ## here, emplicitly inject true labels ##
    #     # # max_scores, pseudo_labels_b = torch.max(predicted_scores, dim=1)
    #     # pseudo_labels_b = (predicted_scores > self.threshold).float() # [B,C]
    #     # predicted_scores = (torch.sigmoid(output) > self.threshold).float()
    #     # pseudo_labels_b = ((predicted_scores + partial_Y - 1) > 0).float() 

    #     predicted_scores = partial_Y 
    #     pseudo_labels_b = (predicted_scores > self.threshold).long()
    #     # print('0', output)
    #     # print('1', partial_Y)
    #     # print('2', pseudo_labels_b)
    #     # using partial labels to filter out negative labels

    #     '''calculate CLS-2 from prototypes'''
    #     # compute protoypical logits
    #     prototypes = self.prototypes.clone().detach() #[C+1,D]
    #     prototype_normal = prototypes[0].unsqueeze(0) # [1,D]
    #     prototypes_abnormal = prototypes[1:] # [C,D]
    #     score_prot = torch.zeros((q.shape[0], self.num_class)).to(q.device) #[B,C]
    #     for i in range(self.num_class):
    #         prototype_i = torch.cat([prototype_normal, prototypes_abnormal[i].unsqueeze(0)], dim=0)
    #         logits_prot_i = torch.mm(q, prototype_i.t()) #[B,2]
    #         score_prot_i = torch.softmax(logits_prot_i, dim=1)[:,1]
    #         score_prot[:,i] = score_prot_i

    #     '''Update prototypes with CLS-1 (and maybe [true label])'''
    #     # update momentum prototypes with pseudo labels
    #     # q: [B,D]; pseudo_labels_b: [B,C]; self.prototypes: [C+1,D]
    #     self._update_prototypes(concat_all_gather(q), concat_all_gather(pseudo_labels_b), args)
      
      
    #     # compute key features 
    #     with torch.no_grad():  # no gradient 
    #         self._momentum_update_key_encoder(args)  # update the momentum encoder
    #         # shuffle for making use of BN
    #         im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)
    #         k, _ = self.encoder_k(im_k)
    #         # undo shuffle
    #         k = self._batch_unshuffle_ddp(k, idx_unshuffle)

    #     features = torch.cat((q, k, self.queue.clone().detach()), dim=0)
    #     pseudo_labels = torch.cat((pseudo_labels_b, pseudo_labels_b, self.queue_pseudo.clone().detach()), dim=0)
    #     # to calculate SupCon Loss using pseudo_labels
        
    #     '''Update queue with CLS-1 (and maybe [true label])'''''
    #     # dequeue and enqueue
    #     self._dequeue_and_enqueue(k, pseudo_labels_b, args)


    #     return features, pseudo_labels, score_prot

    # def _update_prototypes(self, q, pseudo_labels_b, args):
    #     """
    #     原型更新函数
    #     :param q: 当前批次特征 [B, D]
    #     :param pseudo_labels_b: 伪标签矩阵 [B, C]
    #     :param args.proto_m: 动量系数 (0.9)
    #     """
    #     # --------------------------------------------------
    #     # 参数检查
    #     # --------------------------------------------------
    #     assert q.dim() == 2, "特征q必须是二维张量 [B, D]"
    #     assert pseudo_labels_b.dim() == 2, "伪标签必须是二维张量 [B, C]"
    #     B, C = pseudo_labels_b.shape
    #     device = q.device
        
    #     # --------------------------------------------------
    #     # 更新背景原型（索引0）
    #     # --------------------------------------------------
    #     # 筛选无标签样本 (全0行)
    #     bg_mask = (pseudo_labels_b.sum(dim=1) == 0)  # [B]
        
    #     if bg_mask.sum() > 0:
    #         bg_feats = q[bg_mask]  # [num_bg, D]
    #         # 动量更新背景原型
    #         self.prototypes[0] = (
    #             self.prototypes[0] * args.proto_m + 
    #             (1 - args.proto_m) * bg_feats.mean(dim=0)
    #         )

    #     # --------------------------------------------------
    #     # 更新类别原型（索引1~C）
    #     # --------------------------------------------------
    #     for c in range(C):  # 遍历每个标签类别 (c: 0~C-1对应原型的1~C)
    #         # 筛选当前类别的正样本
    #         c_mask = pseudo_labels_b[:, c].bool()  # [B]
    #         if not c_mask.any():
    #             continue  # 无正样本则跳过
            
    #         # --------------------------------------------------
    #         # 核心：计算加权特征均值
    #         # --------------------------------------------------
    #         # 1. 获取正样本特征 [num_pos, D]
    #         c_feats = q[c_mask]
            
    #         # 2. 计算每个样本的权重 (1/该样本的标签数) [num_pos]
    #         with torch.no_grad():
    #             sample_weights = 1.0 / pseudo_labels_b[c_mask].sum(dim=1)  # 归一化权重
                
    #         # 3. 加权平均 (防止除零)
    #         weighted_sum = (c_feats * sample_weights.unsqueeze(1)).sum(dim=0)  # [D]
    #         total_weight = sample_weights.sum()
            
    #         # 4. 动量更新（处理极小权重情况）
    #         if total_weight > 1e-6:
    #             c_proto_idx = c + 1  # 原型索引偏移（因为0是背景）
    #             new_mean = weighted_sum / total_weight
    #             self.prototypes[c_proto_idx] = (
    #                 self.prototypes[c_proto_idx] * args.proto_m + 
    #                 (1 - args.proto_m) * new_mean
    #             )

    #     self.prototypes = F.normalize(self.prototypes, p=2, dim=-1).detach()
    #     return None

# utils
@torch.no_grad()
def concat_all_gather(tensor):
    world_size = dist.get_world_size()
    # rank = dist.get_rank()
    device = tensor.device

    # 收集各进程的张量长度
    local_length = torch.tensor(tensor.size(0), device=device)
    lengths = [torch.empty_like(local_length) for _ in range(world_size)]
    dist.all_gather(lengths, local_length)
    lengths = [int(l.item()) for l in lengths]
    max_length = max(lengths)

    # 填充当前张量到最大长度
    padded_tensor = torch.zeros(
        (max_length, *tensor.shape[1:]), 
        dtype=tensor.dtype, 
        device=device
    )
    padded_tensor[:tensor.size(0)] = tensor  # 填充有效数据部分

    # 收集所有填充后的张量
    padded_tensors = [torch.zeros_like(padded_tensor) for _ in range(world_size)]
    dist.all_gather(padded_tensors, padded_tensor)

    # 拼接有效数据（去除填充部分）
    output = torch.cat([
        padded_tensor_i[:length_i] 
        for padded_tensor_i, length_i in zip(padded_tensors, lengths)
    ], dim=0)

    return output
# def concat_all_gather(local_tensor):
#     device = dist.get_rank()
#     world_size = dist.get_world_size()
#     # Step 1: 获取各 rank 的长度
#     local_len = len(local_tensor)
#     length_tensor = torch.tensor([local_len], device=device)

#     gathered_lengths = [torch.tensor([0], device=device) for _ in range(world_size)]
#     dist.all_gather(gathered_lengths, length_tensor)
#     print(gathered_lengths)
#     max_len = max([t.item() for t in gathered_lengths])

#     # Step 2: 填充本地 tensor
#     padded_tensor = torch.nn.functional.pad(
#         local_tensor,
#         (0, max_len - local_len),
#         value=0
#     )

#     # Step 3: all_gather
#     gathered_tensors = [torch.zeros_like(padded_tensor) for _ in range(world_size)]
#     dist.all_gather(gathered_tensors, padded_tensor)

#     # Step 4: 去掉填充
#     result = [t[:length] for t, length in zip(gathered_tensors, [l.item() for l in gathered_lengths])]

#     return torch.cat(result)
# def concat_all_gather(tensor):
#     """
#     Performs all_gather operation on the provided tensors.
#     *** Warning ***: torch.distributed.all_gather has no gradient.
#     """
#     tensors_gather = [torch.ones_like(tensor)
#         for _ in range(torch.distributed.get_world_size())]
#     torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

#     output = torch.cat(tensors_gather, dim=0)
#     return output

class PolypPVT_CLS(nn.Module):
    def __init__(self, num_class, pretrained=False, feature_dim=512,multi_label=False):
        super(PolypPVT_CLS, self).__init__()

        self.backbone = pvt_v2_b2()  # [64, 128, 320, 512]
        if pretrained:
            path = './pretrained/PolypPVT.pth'
            print('Pretrained from PolypPVT.pth')
            save_model = torch.load(path)
            model_dict = self.backbone.state_dict()
            state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
            model_dict.update(state_dict)
            self.backbone.load_state_dict(model_dict,strict=False)


        print('add cls head')
        if not multi_label:
            self.cls_head = nn.Linear(feature_dim, num_class)  #  → [B, C]
        else:
            self.cls_head = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 8),
        )

    def forward(self, x):

        # backbone
        feature = self.backbone(x)
        # clshead
        logits = self.cls_head(feature)
        return logits
