import torch

if __name__ == "__main__":
    device = "cuda"
    # Conv3D speed, based on kernel size
    x = torch.rand(1,256,64,64,64, device=device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)   
    sizes = (1,3,5)
    for size in sizes:
        # dry run
        a = torch.nn.Conv3d(256, 1, size, padding=(size)//2).to(device)
        b = a(x)
        torch.cuda.synchronize()

        # measured run
        start.record()
        b = a(x)
        end.record()
        torch.cuda.synchronize()
        print(x.shape, "->", b.shape)
        print("Kernel Size:", size, "->", start.elapsed_time(end))

    x = torch.rand(1,3,64,64,64, device=device)
    x = x.reshape(-1, 3)
    a = torch.nn.Linear(3, 1).to(device)
    b = a(x)
    torch.cuda.synchronize()
    start.record()
    b = a(x)
    end.record()
    torch.cuda.synchronize()
    print(x.shape, "->", b.shape)
    print("Linear:", start.elapsed_time(end))
    
    # Interpolation
    x = torch.rand(1,256,32,32,32, device=device)
    a = torch.nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
    y = a(x)
    torch.cuda.synchronize()
    start.record()
    b = a(x)
    end.record()
    torch.cuda.synchronize()
    print(x.shape, "->", b.shape)
    print("Upsample tri:", start.elapsed_time(end))
    
    # Deconv
    x = torch.rand(1,256,32,32,32, device=device)
    a = torch.nn.ConvTranspose3d(256,1,3,stride=2).to(device)
    y = a(x)
    torch.cuda.synchronize()
    start.record()
    b = a(x)
    end.record()
    torch.cuda.synchronize()
    print(x.shape, "->", b.shape)
    print("Deconv:", start.elapsed_time(end))
