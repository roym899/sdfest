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

    print(filename)
    lines = [line.rstrip('\n') for line in open(filename)]
    for line in lines:
        line_split = line.split(' ')
        object_name.append(line_split[0])
        image_id.append(line_split[1])
        t.append([float(line_split[2]), float(line_split[3]), float(line_split[4])])
        q.append([float(line_split[5]), float(line_split[6]), float(line_split[7]), float(line_split[8])])

    return (image_id, object_name, np.asarray(t), np.asarray(q))

if __name__ == "__main__":
    args = parse_args()
    print(args)

    all_score = []
    all_score25 = []
    all_rot_err = []
    all_trans_err = []

    # get GT and estimated pose files
    folder = './results/{}/{}/'.format(args.result_folder, args.obj_ctg)
    files_est = sorted(glob.glob(folder+"*/Pose_{}_*.txt".format(args.obj_ctg)))
    files_gt = sorted(glob.glob(folder + "*/Pose_GT_{}_*.txt".format(args.obj_ctg)))

    # histogram
    rot_err_his = []
    trans_err_his = []

    for file_est, file_gt in zip(files_est, files_gt):
        print("********************************************************")
        print('Start evaluating: {}'.format(file_est))
        (img_ids_est, _, t_est, q_est) = load_data(file_est)
        (img_ids_gt, _, t_gt, q_gt) = load_data(file_gt)

        score = 0
        score_25 = 0
        rot_err = 0
        trans_err = 0

        cls_in_5_5 = 0
        cls_iou_25 = 0

        cls_rot = []
        cls_trans = []

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
            if miou > 0.25:
                cls_iou_25 = cls_iou_25 + 1
            if result[0] < 5 and result[1] < 50:
                cls_in_5_5 = cls_in_5_5 + 1

        n_data_point = float(len(img_ids_est))
        score = cls_in_5_5 / n_data_point
        score_25 = cls_iou_25 / n_data_point
        rot_err = np.mean(cls_rot)
        trans_err = np.mean(cls_trans)
        print('number of data point: ', n_data_point)
        print("5cm 5degree:", score * 100)
        print("IoU 25:     ", score_25 * 100)
        print("rot error:  ", rot_err)
        print("tran error: ", trans_err)
        all_score.append(score)
        all_score25.append(score_25)
        all_rot_err.append(rot_err)
        all_trans_err.append(trans_err/10)
        print('Finish evaluating: {} !'.format(file_est))
        print("********************************************************")

    print("********************************************************")
    print("Overall Mean 5cm 5degree:", np.mean(np.array(all_score) * 100))
    print("Overall Mean IoU 25:     ", np.mean(np.array(all_score25) * 100))
    print("Overall Mean rot error:  ", np.mean(np.array(all_rot_err)))
    print("Overall Mean tran error: ", np.mean(np.array(all_trans_err)))
    print("********************************************************")

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