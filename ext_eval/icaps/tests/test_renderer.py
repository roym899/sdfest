from ycb_render.shapenet_renderer_tensor import *

if __name__ == '__main__':
    model_path = sys.argv[1]
    w = 640
    h = 480

    renderer = ShapeNetTensorRenderer(w, h, render_rotation=False)
    models = [
        "0003",
    ]
    obj_paths = ['{}/camera/{}/model.ply'.format(model_path, item) for item in models]
    texture_paths = [''.format(model_path, item) for item in models]
    renderer.load_objects(obj_paths, texture_paths)

    # mat = pose2mat(pose)
    pose = np.array([0, 0, 0, 1, 0, 0, 0])
    # pose2 = np.array([-0.56162935, 0.05060109, -0.028915625, 0.6582951, 0.03479896, -0.036391996, -0.75107396])
    # pose3 = np.array([0.22380374, 0.019853603, 0.12159989, -0.40458265, -0.036644224, -0.6464779, 0.64578354])

    theta = 0
    phi = 0
    psi = 0
    r = 1.2
    cam_pos = [np.sin(theta) * np.cos(phi) * r, np.sin(phi) * r, np.cos(theta) * np.cos(phi) * r]
    renderer.set_camera(cam_pos, [0, 0, 0], [0, 1, 0])
    renderer.set_fov(40)
    renderer.set_poses([pose])
    renderer.set_light_pos(cam_pos)
    renderer.set_light_color([1.5, 1.5, 1.5])
    tensor = torch.cuda.ByteTensor(h, w, 4)
    tensor2 = torch.cuda.ByteTensor(h, w, 4)

    tensor = torch.cuda.FloatTensor(h, w, 4)
    tensor2 = torch.cuda.FloatTensor(h, w, 4)
    pc_tensor = torch.cuda.FloatTensor(h, w, 4)
    normal_tensor = torch.cuda.FloatTensor(h, w, 4)

    while True:
        # renderer.set_light_pos([0,-1 + 0.01 * i, 0])
        renderer.render([0], tensor, tensor2, pc1_tensor=pc_tensor, pc2_tensor=normal_tensor)

        img_np = tensor.flip(0).data.cpu().numpy().reshape(h, w, 4)
        img_np2 = tensor2.flip(0).data.cpu().numpy().reshape(h, w, 4)
        img_normal = pc_tensor.flip(0).data.cpu().numpy().reshape(h, w, 4)

        img_disp =img_normal[:, :, 0] * 5
        img_disp[img_disp>0] += 0.2

        if len(sys.argv) > 2 and sys.argv[2] == 'headless':
            # print(np.mean(frame[0]))
            theta += 0.001
            if theta > 1: break
        else:
            cv2.imshow('test', cv2.cvtColor(img_disp, cv2.COLOR_RGB2BGR))
            q = cv2.waitKey(16)
            if q == ord('w'):
                phi += 0.1
            elif q == ord('s'):
                phi -= 0.1
            elif q == ord('a'):
                theta -= 0.1
            elif q == ord('d'):
                theta += 0.1
            elif q == ord('n'):
                r -= 0.1
            elif q == ord('m'):
                r += 0.1
            elif q == ord('p'):
                Image.fromarray((img_np[:, :, :3] * 255).astype(np.uint8)).save('test.png')
            elif q == ord('q'):
                break

            # print(renderer.get_poses())
        cam_pos = [np.sin(theta) * np.cos(phi) * r, np.sin(phi) * r, np.cos(theta) * np.cos(phi) * r]
        renderer.set_camera(cam_pos, [0, 0, 0], [0, 1, 0])
        renderer.set_light_pos(cam_pos)

    renderer.release()