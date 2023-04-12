import os
import pickle
import torch
import numpy as np
import zmq
from models import GCN_class_simple, GCN_class, GCN_corr, GCN_corr_class, GCN_corr_class_ours
from opt import Options, setup_folder, save_opt
from utils import *
from data.skeleton_uniform import params,centralize_normalize_rotate_poses

def preprocess_data(poses_gt):
    dct_n = 25
    joints = list(range(15)) + [19, 21, 22, 24]

    with open("data/EC3D/data_3D.pickle", "rb") as f:
            data_3D = pickle.load(f)
    joint_names, subject_1_pose, bone_connections = params(data_3D)
    pose_dict = {'joints':joint_names,'default':subject_1_pose,'links':bone_connections}
    poses_gt = torch.from_numpy(poses_gt)
    poses_gt = centralize_normalize_rotate_poses(poses_gt,pose_dict).numpy()
    
    poses_gt = poses_gt[:, :, joints]
    inputs_raw = [poses_gt.reshape(-1, poses_gt.shape[1] * poses_gt.shape[2]).T]
    inputs = [dct_2d(torch.from_numpy(x))[:, :dct_n].numpy() if x.shape[1] >= dct_n else
                       dct_2d(torch.nn.ZeroPad2d((0, dct_n - x.shape[1], 0, 0))(torch.from_numpy(x))).numpy()
                       for x in inputs_raw]
    
    inputs = torch.from_numpy(np.asarray(inputs))
    inputs = inputs.cuda().float()

    return inputs_raw, inputs

def get_result(opt,model,poses_gt):
    use_random_one_hot = False
    labels = [] # not used
    with torch.no_grad():

        inputs_raw, inputs = preprocess_data(poses_gt)

        deltas, _, y_pred = model(inputs, labels, Use_label=False,random_one_hot=use_random_one_hot)
        _, pred_in = torch.max(y_pred.data, 1)  
        
        outputs = inputs+deltas
        _, pred_out = torch.max(model(outputs, labels, Use_label=False, random_one_hot=use_random_one_hot)[2].data, 1)
        for i, o in enumerate(inputs_raw):
            length = o.shape[1]
            org_raw = torch.from_numpy(o).T*3000
            label = pred_in

            if length > outputs[i].shape[1]:
                m = torch.nn.ZeroPad2d((0, length - deltas[i].shape[1], 0, 0))
                # delt = dct.idct_2d(m(deltas[i]).T.unsqueeze(0))
                outputs_raw = idct_2d(m(outputs[i].cpu()).T.unsqueeze(0))[0]*3000
            else:
                # delt = dct.idct_2d(deltas[i, :, :length].T.unsqueeze(0))
                outputs_raw = idct_2d(outputs[i, :, :length].cpu().T.unsqueeze(0))[0]*4000

            for t in range(length):
                # change figure path to Visual
                fig_loc ="Visual/"+opt.datetime
                fig_loc += "/" + str(i) + "_" + str(label.item())
                # if label > 8:
                if not os.path.exists(fig_loc):
                    os.makedirs(fig_loc)
                display_poses([org_raw[t].reshape([3,19])], save_loc=fig_loc, custom_name="outputs_", time=t, custom_title=None, legend_=None, color_list = ["red"])
                display_poses([org_raw[t].reshape([3,19]),outputs_raw[t].reshape([3,19])], save_loc=fig_loc, custom_name="inputs_", time=t, custom_title=None, legend_=None, color_list = ["red", "green"])

def main(opt, model_version):
    is_cuda = torch.cuda.is_available()
    models = {'Separated_Classifier_Simple': GCN_class_simple(hidden_feature=opt.hidden, p_dropout=opt.dropout, classes=12).cuda(),
              'Separated_Classifier': GCN_class(hidden_feature=opt.hidden, p_dropout=opt.dropout, dataset_name='EC3D').cuda(),
              'Separated_Corrector': GCN_corr(hidden_feature=opt.hidden,  p_dropout=opt.dropout).cuda(),
              'Combined_wo_Feedback': GCN_corr_class(hidden_feature=opt.hidden, p_dropout=opt.dropout, classes=12).cuda(),
              'Ours': GCN_corr_class_ours(hidden_feature=opt.hidden, p_dropout=opt.dropout, classes=12).cuda()
              }
    model = models[model_version]
    if is_cuda:
        model.cuda()
    model_path = 'pretrained_weights/Ours.pt'
    print('Use the pre-trained model.')
    opt.model_dir = model_path
    # Test
    model = models[model_version]
    model.load_state_dict(torch.load(opt.model_dir, map_location='cuda:0'))
    model.cuda()

    model_id = model_path[-19:-3] if model_path[0] != 'p' else 'pretrained'

    if is_cuda:
        model.cuda()
        model.eval()

    try:
        with open("data/kinect/data.pickle","rb") as f:
            poses_gt = pickle.load(f)
        print('Loading reserved data.')
        get_result(opt,model,poses_gt)
    except FileNotFoundError:
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.bind("tcp://*:5555")
        count = 1
        while True:
            #  Wait for next request from client
            print("Waiting for data")
            poses_gt = pickle.loads(socket.recv())
            with open("data/kinect/data.pickle","wb") as f:
                pickle.dump(poses_gt,f)
            print(f"Received request: {count}\n {poses_gt}\n")

            get_result(opt,model,poses_gt)
            #  Send reply back to client
            socket.send(b"nothing")
            count = count + 1

    

if __name__ == "__main__":
    torch.cuda.set_device(0)
    print('GPU Index: {}'.format(torch.cuda.current_device()))

    model_version = 'Ours'
    option = Options().parse()
    main(option, model_version)