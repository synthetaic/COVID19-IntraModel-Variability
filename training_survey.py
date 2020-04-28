import os
import argparse
import multiprocessing as mp

parser = argparse.ArgumentParser()
parser.add_argument('--data-docker', default='/data/data/synthetaic/czhong/COVIDClassifier/data_v3', type=str, help='Data directory within the Docker container of the training, val, and testing samples.')
parser.add_argument('--gpu-docker', default='0,1,2,3,4,5,6,7', type=str, help='Comma delimited string of GPUs to use. 1 docker = 1 GPUs')
parser.add_argument('--training-script-docker', default='/data/data/synthetaic/czhong/COVIDClassifier/Covid19DemoV3_final1.py', type=str, help='Training script to parallelize.')

args = parser.parse_args()
docker_gpus= [gpu for gpu in args.gpu_docker.split(',')]
manager = mp.Manager().list()

for gpu in docker_gpus:
    manager.append(gpu)

run_params = [
    ['resnet18-512-1', 'resnet18', args.data_docker, manager, args.training_script_docker],
    ['resnet18-512-2', 'resnet18', args.data_docker, manager, args.training_script_docker],
    ['resnet18-512-3', 'resnet18', args.data_docker, manager, args.training_script_docker],
    ['resnet50-512-1', 'resnet50', args.data_docker, manager, args.training_script_docker],
    ['resnet50-512-2', 'resnet50', args.data_docker, manager, args.training_script_docker],
    ['resnet50-512-3', 'resnet50', args.data_docker, manager, args.training_script_docker],
    ['resnet101-512-1', 'resnet101', args.data_docker, manager, args.training_script_docker],
    ['resnet101-512-2', 'resnet101', args.data_docker, manager, args.training_script_docker],
    ['resnet101-512-3', 'resnet101', args.data_docker, manager, args.training_script_docker],
    ['resnet152-512-1', 'resnet152', args.data_docker, manager, args.training_script_docker],
    ['resnet152-512-2', 'resnet152', args.data_docker, manager, args.training_script_docker],
    ['resnet152-512-3', 'resnet152', args.data_docker, manager, args.training_script_docker],
    ['wideresnet50-512-1', 'wideresnet50', args.data_docker, manager, args.training_script_docker],
    ['wideresnet50-512-2', 'wideresnet50', args.data_docker, manager, args.training_script_docker],
    ['wideresnet50-512-3', 'wideresnet50', args.data_docker, manager, args.training_script_docker],
    ['wideresnet101-512-1', 'wideresnet101', args.data_docker, manager, args.training_script_docker],
    ['wideresnet101-512-2', 'wideresnet101', args.data_docker, manager, args.training_script_docker],
    ['wideresnet101-512-3', 'wideresnet101', args.data_docker, manager, args.training_script_docker],
    ['densenet121-512-1', 'densenet121', args.data_docker, manager, args.training_script_docker],
    ['densenet121-512-2', 'densenet121', args.data_docker, manager, args.training_script_docker],
    ['densenet121-512-3', 'densenet121', args.data_docker, manager, args.training_script_docker],
    ['densenet169-512-1', 'densenet169', args.data_docker, manager, args.training_script_docker],
    ['densenet169-512-2', 'densenet169', args.data_docker, manager, args.training_script_docker],
    ['densenet169-512-3', 'densenet169', args.data_docker, manager, args.training_script_docker],
    ['densenet201-512-1', 'densenet201', args.data_docker, manager, args.training_script_docker],
    ['densenet201-512-2', 'densenet201', args.data_docker, manager, args.training_script_docker],
    ['densenet201-512-3', 'densenet201', args.data_docker, manager, args.training_script_docker],
    ['resnext50_32x4d-512-1', 'resnext50_32x4d', args.data_docker, manager, args.training_script_docker],
    ['resnext50_32x4d-512-2', 'resnext50_32x4d', args.data_docker, manager, args.training_script_docker],
    ['resnext50_32x4d-512-3', 'resnext50_32x4d', args.data_docker, manager, args.training_script_docker],
    ['resnext101_32x8d-512-1', 'resnext101_32x8d', args.data_docker, manager, args.training_script_docker],
    ['resnext101_32x8d-512-2', 'resnext101_32x8d', args.data_docker, manager, args.training_script_docker],
    ['resnext101_32x8d-512-3', 'resnext101_32x8d', args.data_docker, manager, args.training_script_docker],
    ['mobilenet_v2-512-1', 'mobilenet_v2', args.data_docker, manager, args.training_script_docker],
    ['mobilenet_v2-512-2', 'mobilenet_v2', args.data_docker, manager, args.training_script_docker],
    ['mobilenet_v2-512-3', 'mobilenet_v2', args.data_docker, manager, args.training_script_docker],
]

def work(kwargs):
    print('Work started')
    proj_name, arch, data_folder, open_dockers, script = kwargs[0], kwargs[1], kwargs[2], kwargs[3], kwargs[4]
    open_gpu = open_dockers[0]
    open_dockers.remove(open_gpu)
    open_string = 'dockergpu{}'.format(str(open_gpu))
    print('Training {} in {}'.format(proj_name, open_string))
    ckpt_dir = './checkpoints'

    proj_ckpt = os.path.join(ckpt_dir, proj_name)
    if os.path.exists(proj_ckpt) is False:
        os.makedirs(proj_ckpt, exist_ok=True)

    log_dir = os.path.join(proj_ckpt, 'logs')
    if os.path.exists(log_dir) is False:
        os.makedirs(log_dir, exist_ok=True)

    cm_dir = os.path.join(proj_ckpt, 'cms')
    if os.path.exists(cm_dir) is False:
        os.makedirs(cm_dir, exist_ok=True)

    graph_dir = os.path.join(proj_ckpt, 'graphs')
    if os.path.exists(graph_dir) is False:
        os.makedirs(graph_dir, exist_ok=True)

    os.system('docker container start {}'.format(open_string))
    commandline = 'docker exec {} python3 {} --proj-name={} --arch={} --gpus=0 --data={} '.format(open_string, script, proj_name, arch, data_folder)
    os.system(commandline)
    os.system('docker container stop {}'.format(open_string))
    open_dockers.append(open_gpu)   

if __name__ == '__main__':
    
    docker_gpus= [gpu for gpu in args.gpu_docker.split(',')]
    docker_count = len(docker_gpus)
    tasks = len(run_params)
    do = range(tasks)
    pool = mp.Pool(docker_count)

    # Docker containers that are available to be used for training.
    results = pool.map(work, run_params)
    pool.close()

