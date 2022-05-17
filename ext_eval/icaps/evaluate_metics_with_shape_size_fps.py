import matplotlib.pyplot as plt
from datasets.nocs_benchmark import *
import argparse
import glob
from transforms3d.quaternions import *

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Test Pose Estimation Network with Multiple Object Models')
    parser.add_argument('--obj_ctg', dest='obj_ctg',
                        help='test object category',
                        required=True, type=str)
    parser.add_argument('--result_folder', dest='result_folder',
                        help='result folder',
                        required=True, type=str)
    parser.add_argument('--scale_dir', dest='scale_dir',
                        help='directory of the NOCS bboxes',
                        default='../NOCS_dataset/real_test_obj_scale/',
                        type=str)
    args = parser.parse_args()
    return args

def load_data(filename):
    image_id = []
    object_name = []
    t = []
    q = []
    s = []
    fps = []
    print(filename)
    lines = [line.rstrip('\n') for line in open(filename)]
    for line in lines:
        line_split = line.split(' ')
        object_name.append(line_split[0])
        image_id.append(line_split[1])
        t.append([float(line_split[2]), float(line_split[3]), float(line_split[4])])
        q.append([float(line_split[5]), float(line_split[6]), float(line_split[7]), float(line_split[8])])
        s.append(float(line_split[9]))
        fps.append(float(line_split[10]))
    return (image_id, object_name, np.asarray(t), np.asarray(q), np.asarray(s), np.asarray(fps))

def load_shape_data(filename):
    image_id = []
    object_name = []
    chamfer_init = []
    chamfer_opt = []

    print(filename)
    lines = [line.rstrip('\n') for line in open(filename)]
    for line in lines:
        line_split = line.split(' ')
        object_name.append(line_split[0])
        image_id.append(line_split[1])
        chamfer_init.append(float(line_split[2]))
        chamfer_opt.append(float(line_split[3]))

    return (image_id, object_name, np.asarray(chamfer_init), np.asarray(chamfer_opt))

if __name__ == "__main__":
    args = parse_args()
    print(args)

    all_score = []
    all_score25 = []
    all_rot_err = []
    all_trans_err = []
    all_dist_init = []
    all_dist_opt = []
    all_size_err = []
    all_fps = []
    # get GT and estimated pose files
    folder = './results/{}/{}/'.format(args.result_folder, args.obj_ctg)
    files_est = sorted(glob.glob(folder+"*/Pose_{}_*.txt".format(args.obj_ctg)))
    files_gt = sorted(glob.glob(folder + "*/Pose_GT_{}_*.txt".format(args.obj_ctg)))
    files_shape = sorted(glob.glob(folder + "*/Shape_{}_*.txt".format(args.obj_ctg)))

    report = open(folder + "/Evaluate_report_{}.txt".format(args.obj_ctg), "w+")

    # histogram
    rot_err_his = []
    trans_err_his = []
    size_err_his = []

    for file_est, file_gt in zip(files_est, files_gt):
        print("********************************************************")
        print('Start evaluating: {}'.format(file_est))
        report.write('********************************************************\n')
        report.write('Start evaluating: {} \n'.format(file_est))

        (img_ids_est, _, t_est, q_est, s_est, fps) = load_data(file_est)
        (img_ids_gt, _, t_gt, q_gt, s_gt, fps) = load_data(file_gt)

        score = 0
        score_25 = 0
        rot_err = 0
        trans_err = 0

        cls_in_5_5 = 0
        cls_iou_25 = 0

        cls_rot = []
        cls_trans = []
        cls_size = []
        cls_fps = []
        # real model name
        model_name = img_ids_gt[0].split('/')[-1]

        for i in range(len(img_ids_est)):
            assert img_ids_est[i] == img_ids_gt[i], "cannot match gt and estimation files"
            T_est = np.eye(4)
            T_est[:3, :3] = quat2mat(q_est[i])
            T_est[:3, 3] = t_est[i] * 1000

            T_gt = np.eye(4)
            T_gt[:3, :3] = quat2mat(q_gt[i])
            T_gt[:3, 3] = t_gt[i] * 1000

            result = compute_RT_degree_cm_symmetry(T_est, T_gt, args.obj_ctg)
            bbox = np.loadtxt(args.scale_dir + model_name + ".txt").transpose()
            miou = compute_3d_iou_new(T_gt, T_est, bbox, bbox, 1, args.obj_ctg,
                                      args.obj_ctg)

            if miou > 0.25 and result[0] < 360:
                cls_rot.append(result[0])
                rot_err_his.append(result[0])
            if miou > 0.25:
                cls_trans.append(result[1])
                trans_err_his.append(result[1])
                cls_size.append(abs(s_est - s_gt))
                cls_fps.append(fps)
            if miou > 0.25:
                cls_iou_25 = cls_iou_25 + 1
            if result[0] < 5 and result[1] < 50:
                cls_in_5_5 = cls_in_5_5 + 1

        n_data_point = float(len(img_ids_est))
        score = cls_in_5_5 / n_data_point
        score_25 = cls_iou_25 / n_data_point
        rot_err = np.mean(cls_rot)
        trans_err = np.mean(cls_trans)
        size_err = np.mean(cls_size)
        fps_rec = np.mean(cls_fps)
        print('number of data point: ', n_data_point)
        print("5cm 5degree:", score * 100)
        print("IoU 25:     ", score_25 * 100)
        print("rot error:  ", rot_err)
        print("tran error: ", trans_err)
        print("size error: ", size_err)
        print("fps: ", fps_rec)
        all_score.append(score)
        all_score25.append(score_25)
        all_rot_err.append(rot_err)
        all_trans_err.append(trans_err/10)
        all_size_err.append(size_err)
        all_fps.append(fps_rec)
        print('Finish evaluating: {} !'.format(file_est))
        print("********************************************************")
        report.write('number of data point: {}  \n'.format(n_data_point))
        report.write('5cm 5degree:{:.5f} \n'.format(score * 100))
        report.write('IoU 25:     {:.5f} \n'.format(score_25 * 100))
        report.write('rot error:  {:.5f} \n'.format(rot_err))
        report.write('tran error: {:.5f} \n'.format(trans_err))
        report.write('size error: {:.5f} \n'.format(size_err))
        report.write('fps record: {:.5f} \n'.format(fps_rec))
        report.write('Finish evaluating: {} ! \n'.format(file_est))
        report.write('********************************************************\n')


    print("********************************************************")
    print("Overall Mean 5cm 5degree:", np.mean(np.array(all_score) * 100))
    print("Overall Mean IoU 25:     ", np.mean(np.array(all_score25) * 100))
    print("Overall Mean rot error:  ", np.mean(np.array(all_rot_err)))
    print("Overall Mean tran error: ", np.mean(np.array(all_trans_err)))
    print("Overall Mean size error: ", np.mean(np.array(all_size_err)))
    print("Overall Mean fps record: ", np.mean(np.array(all_fps)))
    print("********************************************************")

    report.write('********************************************************\n')
    report.write('Overall Mean 5cm 5degree: {:.5f} \n'.format(np.mean(np.array(all_score) * 100)))
    report.write('Overall Mean IoU 25:      {:.5f} \n'.format(np.mean(np.array(all_score25) * 100)))
    report.write('Overall Mean rot error:   {:.5f} \n'.format(np.mean(np.array(all_rot_err))))
    report.write('Overall Mean tran error:  {:.5f} \n'.format(np.mean(np.array(all_trans_err))))
    report.write('Overall Mean size error:  {:.5f} \n'.format(np.mean(np.array(all_size_err))))
    report.write('Overall Mean fps record:  {:.5f} \n'.format(np.mean(np.array(all_fps))))
    report.write('********************************************************\n')

    # plot the histogram
    rot_err_his = np.array(rot_err_his, dtype=np.float32)
    bins = np.linspace(0, 180, 36)
    weights = np.ones_like(rot_err_his) / rot_err_his.shape[0] * 100
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.hist(rot_err_his, bins, weights=weights)
    ax.axis(np.array([0, 180, 0, 40], dtype=np.float32))
    plt.xlabel('Rotation Error (deg)')
    plt.ylabel('Data Percentage (%)')
    plt.title('Rotation Error Distribution: {}'.format(args.obj_ctg))
    plt.savefig('{}rot_dist_{}.png'.format(folder,args.obj_ctg))
    # plt.show()

    for file_shape in files_shape:
        print("********************************************************")
        print('Start evaluating shape: {}'.format(file_shape))
        report.write('********************************************************\n')
        report.write('Start evaluating shape: {} \n'.format(file_est))

        (img_ids_shape, _, dist_init, dist_opt) = load_shape_data(file_shape)

        dist_init_rec = []
        dist_opt_rec = []

        # real model name
        model_name = img_ids_shape[0].split('/')[-1]

        n_data_point = float(len(img_ids_shape))

        for i in range(len(img_ids_shape)):
            if abs(dist_init[i]) < 10000 and abs(dist_opt[i]) < 10000: 
                dist_init_rec.append(dist_init[i])
                dist_opt_rec.append(dist_opt[i])
            else:
                n_data_point -= 1

        dist_init_mean = np.mean(dist_init_rec)
        dist_opt_mean  = np.mean(dist_opt_rec)
        print('number of data point: ', n_data_point)
        print("chamfer init:", dist_init_mean)
        print("chamfer opt: ", dist_opt_mean)
        all_dist_init.append(dist_init_mean)
        all_dist_opt.append(dist_opt_mean)
        print('Finish evaluating shape: {} !'.format(file_shape))
        print("********************************************************")
        report.write('number of data point: {}  \n'.format(n_data_point))
        report.write('chamfer init:{:.5f} \n'.format(dist_init_mean))
        report.write('chamfer opt: {:.5f} \n'.format(dist_opt_mean))
        report.write('Finish evaluating shape: {} ! \n'.format(file_shape))
        report.write('********************************************************\n')

    print("********************************************************")
    print("Overall Chamfer init:  ", np.mean(np.array(all_dist_init)))
    print("Overall Chamfer opt: ", np.mean(np.array(all_dist_opt)))
    print("********************************************************")
    report.write('********************************************************\n')
    report.write('Overall Chamfer init: {:.5f} \n'.format(np.mean(np.array(all_dist_init))))
    report.write('Overall Chamfer opt:  {:.5f} \n'.format(np.mean(np.array(all_dist_opt))))
    report.write('********************************************************\n')
