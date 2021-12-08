import torch
from torchvision.models import AlexNet
from torchvision.models import resnet18
from torchviz import make_dot


if __name__ == "__main__":
    x = torch.rand(1, 3, 256, 256)
    # model = AlexNet()
    model = resnet18()
    y = model(x)

    # 这三种方式都可以
    g = make_dot(y)
    # g=make_dot(y, params=dict(model.named_parameters()))
    # g = make_dot(y, params=dict(list(model.named_parameters()) + [('x', x)]))


    # 这两种方法都可以
    g.view() # 会生成一个 Digraph.gv.pdf 的PDF文件
    # g.render('espnet_model', view=True)  # 会自动保存为一个 espnet.pdf，第二个参数为True,则会自动打开该PDF文件，为False则不打开
    print(model)
    params = list(model.parameters())
    k = 0
    for i in params:
        l = 1
        print("该层的结构：" + str(list(i.size())))
        for j in i.size():
            l *= j
        print("该层参数和：" + str(l))
        k = k + l
    print("总参数数量和：" + str(k))