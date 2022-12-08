lstsp_size=$1
n_clusters=$2
sub_graph_size=$3
pretrained_model_path=$4
python train_path_solver.py graph_size=$sub_graph_size val_type='x8Aug_2Traj' \
    pretrained_model_path=$pretrained_model_path
python train_loop_solver.py graph_size=$n_clusters lstsp_size=$lstsp_size \
    pretrained_model_path=$pretrained_model_path
python eval.py pretrained_model_path=$pretrained_model_path