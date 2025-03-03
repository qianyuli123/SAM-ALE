# import torch
# import torch.nn as nn
# from transformers import ViTModel

# class TransformerBlock(nn.Module):
#     def __init__(self, embed_dim, num_heads, mlp_ratio):
#         super(TransformerBlock, self).__init__()
#         self.attn = nn.MultiheadAttention(embed_dim, num_heads)
#         self.mlp = nn.Sequential(
#             nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
#             nn.GELU(),
#             nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
#         )
#         self.norm1 = nn.LayerNorm(embed_dim)
#         self.norm2 = nn.LayerNorm(embed_dim)

#     def forward(self, x):
#         # Multi-head Self-Attention
#         attn_out, _ = self.attn(x, x, x)  # Self-attention
#         x = x + attn_out  # Residual connection
#         x = self.norm1(x)

#         # Feedforward MLP
#         mlp_out = self.mlp(x)
#         x = x + mlp_out  # Residual connection
#         x = self.norm2(x)

#         return x

# class CustomVisionTransformer(nn.Module):
#     def __init__(self, img_size=1024, patch_size=16, in_chans=3, embed_dim=576, num_heads=8, depth=12, mlp_ratio=4.0, pretrained_model_path=None):
#         super(CustomVisionTransformer, self).__init__()
#         self.img_size = img_size
#         self.patch_size = patch_size
#         self.num_patches = (img_size // patch_size) * (img_size // patch_size)
#         self.embed_dim = embed_dim

#         # Patch embedding
#         self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        
#         # Position embeddings
#         self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
#         self.cls_token = None  # 不使用分类 token

#         # Transformer blocks
#         self.transformer_blocks = nn.ModuleList([TransformerBlock(embed_dim, num_heads, mlp_ratio) for _ in range(depth)])

#         # LayerNorm for final features
#         self.norm = nn.LayerNorm(embed_dim)

#         # 如果指定了预训练模型路径，则加载预训练权重
#         if pretrained_model_path:
#             self.load_pretrained_weights(pretrained_model_path)

#     def load_pretrained_weights(self, pretrained_model_path):
#         """加载预训练模型的权重"""
#         pretrained_model = ViTModel.from_pretrained(pretrained_model_path)
        
#         # 获取预训练模型的权重字典
#         pretrained_dict = pretrained_model.state_dict()
#         model_dict = self.state_dict()

#         # 获取预训练模型和自定义模型中匹配的部分
#         pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.shape == model_dict[k].shape}

#         # 加载匹配的部分
#         model_dict.update(pretrained_dict)

#         # 加载自定义模型中未匹配的部分（初始化它们）
#         # 加载自定义模型中未匹配的部分（初始化它们）
#         for name, param in model_dict.items():
#             if name not in pretrained_dict:
#                 # 检查参数的维度，如果是二维及以上，使用xavier_uniform初始化
#                 if param.dim() > 1:
#                     nn.init.xavier_uniform_(param)  # Xavier初始化其他层
#                 else:
#                     nn.init.normal_(param, mean=0.0, std=0.02)  # 如果是1维或0维，使用正态分布初始化

#         # 最后加载权重
#         self.load_state_dict(model_dict)
        # def forward(self, x):
        #     # Step 1: Patch embedding
        #     x = self.patch_embed(x)  # Shape: [batch_size, embed_dim, H_patches, W_patches]
        #     x = x.flatten(2).transpose(1, 2)  # Flatten patches into tokens; Shape: [batch_size, num_patches, embed_dim]

        #     # Step 2: Add position embeddings
        #     x = x + self.pos_embed

        #     # Step 3: Pass through Transformer blocks
        #     for block in self.transformer_blocks:
        #         x = block(x)

        #     # Step 4: Normalize features
        #     x = self.norm(x)

        #     return x  # Final shape: [batch_size, num_patches, embed_dim]
import torch
import torch.nn as nn
from transformers import ViTModel

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio):
        super(TransformerBlock, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # Multi-head Self-Attention
        attn_out, _ = self.attn(x, x, x)  # Self-attention
        x = x + attn_out  # Residual connection
        x = self.norm1(x)

        # Feedforward MLP
        mlp_out = self.mlp(x)
        x = x + mlp_out  # Residual connection
        x = self.norm2(x)

        return x

class CustomVisionTransformer(nn.Module):
    def __init__(self, img_size=1024, patch_size=16, in_chans=3, embed_dim=768, num_heads=8, depth=12, mlp_ratio=4.0, pretrained_model_path=None):
        super(CustomVisionTransformer, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.embed_dim = embed_dim

        # Patch embedding
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # Position embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        self.cls_token = None  # 不使用分类 token

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([TransformerBlock(embed_dim, num_heads, mlp_ratio) for _ in range(depth)])

        # LayerNorm for final features
        self.norm = nn.LayerNorm(embed_dim)

        # 如果指定了预训练模型路径，则加载预训练权重
        if pretrained_model_path:
            self.load_pretrained_weights(pretrained_model_path)

    def load_pretrained_weights(self, pretrained_model_path):
        """加载预训练模型的权重"""
        pretrained_model = ViTModel.from_pretrained(pretrained_model_path)
        
        # 获取预训练模型的权重字典
        pretrained_dict = pretrained_model.state_dict()
        model_dict = self.state_dict()

        # 获取预训练模型和自定义模型中匹配的部分
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.shape == model_dict[k].shape}

        # 加载匹配的部分
        model_dict.update(pretrained_dict)

        # 加载自定义模型中未匹配的部分（初始化它们）
        # 加载自定义模型中未匹配的部分（初始化它们）
        for name, param in model_dict.items():
            if name not in pretrained_dict:
                # 检查参数的维度，如果是二维及以上，使用xavier_uniform初始化
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param)  # Xavier初始化其他层
                else:
                    nn.init.normal_(param, mean=0.0, std=0.02)  # 如果是1维或0维，使用正态分布初始化

        # 最后加载权重
        self.load_state_dict(model_dict)

    def forward(self, x):
        # Step 1: Patch embedding
        x = self.patch_embed(x)  # Shape: [batch_size, embed_dim, H_patches, W_patches]
        x = x.flatten(2).transpose(1, 2)  # Flatten patches into tokens; Shape: [batch_size, num_patches, embed_dim]

        # Step 2: Add position embeddings
        x = x + self.pos_embed

        # Step 3: Pass through Transformer blocks
        for block in self.transformer_blocks:
            x = block(x)

        # Step 4: Normalize features
        x = self.norm(x)

        x = x.reshape(x.shape[0], -1, 64, 64)
        return x  # Final shape: [batch_size, num_patches, embed_dim]


# 计算模型的参数量
def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

if __name__ == "__main__":
    # 创建自定义模型
    model = CustomVisionTransformer(img_size=1024, patch_size=16, embed_dim=768, num_heads=8, depth=12, mlp_ratio=4.0)

    # 打印模型结构（可选）
    print(model)
    # 定义输入张量 [Batch size, Channels, Height, Width]
    input_tensor = torch.randn(6, 3, 1024, 1024)

    # 前向传播
    output = model(input_tensor)

    # 输出维度
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")  # Expected: [6, 4096, 387]

    print(f"模型的参数量是：{count_parameters(model)}")

