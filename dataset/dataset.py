import torch

class Dataset(torch.utils.data.Dataset):
    """Basic dataset interface"""
    def __init__(self): 
        super().__init__()  #通过调用 super().__init__()，子类可以继承并执行父类的初始化代码，从而获得父类的属性和方法

    def __len__(self):
        raise NotImplementedError   #某个方法或功能尚未被实现或定义，使用 raise NotImplementedError 来抛出异常，以提醒开发者需要实现或重写该部分的代码

    def __getitem__(self):
        raise NotImplementedError

    def collate(self, batch):
        iter_res, iter_spp = batch[0]['resolution'], batch[0]['spp']    #从 batch 中的第一个元素中获取 'resolution' 和 'spp' 字段的值，并分别赋给 iter_res 和 iter_spp 这两个变量
        return {
            'mv' : torch.cat(list([item['mv'] for item in batch]), dim=0),  #在 batch 列表中的每个元素（字典）中获取键为 'mv' 的值，形成一个新的列表
            'mvp' : torch.cat(list([item['mvp'] for item in batch]), dim=0),
            'campos' : torch.cat(list([item['campos'] for item in batch]), dim=0),
            'resolution' : iter_res,
            'spp' : iter_spp,
            'img' : torch.cat(list([item['img'] for item in batch]), dim=0)
        }   #return 语句返回了一个字典，其中包含了多个键值对