import torch
import torch.nn.functional as F

class MSE(torch.nn.Module):
    def __init__(self):
        super(MSE, self).__init__()
        self.mse = self.mse = torch.nn.MSELoss(reduce=True, size_average=True)

    def forward(self, embedding_1, embedding_2):
        return self.mse(embedding_1, embedding_2)

class InfoNCE(torch.nn.Module):
    def __init__(self, temperature):
        super(InfoNCE, self).__init__()
        self.t = temperature

    def forward(self, view1, view2):
        view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
        pos_score = (view1 * view2).sum(dim=-1)
        pos_score = torch.exp(pos_score / self.t)
        ttl_score = torch.matmul(view1, view2.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / self.t).sum(dim=1)
        cl_loss = -torch.log(pos_score / ttl_score)
        return torch.mean(cl_loss)
class MinMI(torch.nn.Module):
    def __init__(self, temperature):
        super(MinMI, self).__init__()
        self.t = temperature

    def forward(self, view1, view2):
        view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
        pos_score = (view1 * view2).sum(dim=-1)
        pos_score = torch.exp(pos_score / self.t)
        ttl_score = torch.matmul(view1, view2.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / self.t).sum(dim=1)
        cl_loss = -torch.log(pos_score / ttl_score)
        return torch.mean(cl_loss)
# class InfoNCE_wo_neg(torch.nn.Module):
#     def __init__(self):
#         super(InfoNCE_wo_neg, self).__init__()

#     def forward(self, view1, view2):
#         view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
#         pos_score = (view1 * view2).sum(dim=-1)
#         # pos_score = torch.exp(pos_score / self.t)
#         ttl_score = torch.matmul(view1, view2.transpose(0, 1))
#         ttl_score = torch.exp(ttl_score / self.t).sum(dim=1)
#         cl_loss = -torch.log(pos_score / ttl_score)
#         return torch.mean(cl_loss)

class BPR(torch.nn.Module):
    def __init__(self):
        super(BPR, self).__init__()

    def forward(self, user_embedding, item_embedding, u, i, j, user_embedding_ego=None, item_embedding_ego=None):
        user = user_embedding[u]
        pos_item = item_embedding[i]
        neg_item = item_embedding[j]
        assert torch.isnan(user).sum() == 0
        assert torch.isnan(pos_item).sum() == 0
        assert torch.isnan(neg_item).sum() == 0
        # -------- BPR loss

        # prediction_i = (user * pos_item).sum(dim=-1)
        # prediction_j = (user * neg_item).sum(dim=-1)
        # assert torch.isnan(prediction_i).sum() == 0
        # assert torch.isnan(prediction_j).sum() == 0

        # bpr_loss = -((prediction_i - prediction_j).sigmoid().log().mean())
        pos_scores = torch.mul(user, pos_item)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(user, neg_item)
        neg_scores = torch.sum(neg_scores, dim=1)
        bpr_loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))

        if user_embedding_ego==None or item_embedding_ego==None:
            reg_loss = ((user ** 2).sum(dim=-1) + (pos_item ** 2 + neg_item ** 2).sum(dim=-1)).mean()
        else:
            userEmb0 = user_embedding_ego[u]
            posEmb0 = item_embedding_ego[i]
            negEmb0 = item_embedding_ego[j]
            reg_loss = (1 / 2) * (userEmb0.norm(2).pow(2) +
                              posEmb0.norm(2).pow(2) +
                              negEmb0.norm(2).pow(2)) / float(len(u))

        return bpr_loss, reg_loss

class Normalize(torch.nn.Module):
    """normalization layer"""

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power) + 1e-8
        norm = norm.sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out
