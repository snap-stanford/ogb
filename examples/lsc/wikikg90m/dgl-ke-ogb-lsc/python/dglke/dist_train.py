# -*- coding: utf-8 -*-
#
# dist_train.py
#
# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
import stat
import sys
import subprocess
import argparse
import socket
if os.name != 'nt':
    import fcntl
    import struct

from .utils import CommonArgParser

SCRIPT_FILE = 'dglke_start_kvserver_kvclient.sh'

class ArgParser(CommonArgParser):
    def __init__(self):
        super(ArgParser, self).__init__()

        self.add_argument('--path', type=str, help='path of distributed workspace.')
        self.add_argument('--ssh_key', type=str, help='ssh private key.')
        self.add_argument('--ip_config', type=str, help='IP configuration file of kvstore.')
        self.add_argument('--num_client_proc', type=int, default=1,
                          help='Number of client process on each machine.')

def get_machine_count(ip_config):
    """Get total machine count from ip_config file
    """
    with open(ip_config) as f:
        machine_count = len(f.readlines())

    return machine_count

def local_ip4_addr_list():
    """Return a set of IPv4 address
    """
    nic = set()

    for ix in socket.if_nameindex():
        name = ix[1]
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        ip = socket.inet_ntoa(fcntl.ioctl(
            s.fileno(),
            0x8915,  # SIOCGIFADDR
            struct.pack('256s', name[:15].encode("UTF-8")))[20:24])
        nic.add(ip)

    return nic

def is_local(ip_addr):
    """If ip_addr is a local ip
    """
    if ip_addr in local_ip4_addr_list():
        return True
    else:
        return False

def run_cmd(cmd_str):
    """run command
    """
    os.environ['PATH'] = '/usr/local/bin:/bin:/usr/bin:/sbin/'
    process = subprocess.Popen(cmd_str, shell=True, env=os.environ)
    return process

def wait_job(process, cmd_str):
    """Wait process finish its job
    """
    retcode = process.wait()
    mesg = 'Fail with retcode(%s): %s' %(retcode, cmd_str)
    if retcode != 0:
        raise RuntimeError(mesg)

def ssh_cmd(cmd_str, ip, ssh_key=None):
    """construct an ssh command
    """
    if ssh_key is None:
        ssh_cmd_str = 'ssh %s \'%s\'' % (ip, cmd_str)
    else:
        ssh_cmd_str = 'ssh -i %s %s \'%s\'' % (ssh_key, ip, cmd_str)

    return ssh_cmd_str

def scp_file(file, ip, path, ssh_key=None):
    """scp file to remote machine
    """
    if ssh_key is None:
        scp_cmd_str = 'scp %s %s:%s' % (file, ip, path)
    else:
        scp_cmd_str = 'scp -i %s %s %s:%s' % (ssh_key, file, ip, path)

    return scp_cmd_str

def construct_cmd_script(args):
    """Construct command line string and write it into file
    """
    total_client = args.num_client_proc * get_machine_count(args.ip_config)
    cmd_str = '''#!/bin/bash
SERVER_ID_LOW=$1
SERVER_ID_HIGH=$2

if [ -f "entity_emb-data-shape" ]; then
    echo "Delete temp files..."
    rm *-shape
fi

while [ $SERVER_ID_LOW -lt $SERVER_ID_HIGH ]
do
    MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 DGLBACKEND=pytorch dglke_server --model %s \
    --dataset %s --data_path %s --format %s --ip_config %s --hidden_dim %d --gamma %f --lr %f \
    --total_client %d --server_id $SERVER_ID_LOW &
    let SERVER_ID_LOW+=1
done

MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 DGLBACKEND=pytorch dglke_client --model %s \
--dataset %s --data_path %s --format %s --save_path %s --ip_config %s --batch_size %d \
--neg_sample_size %d --hidden_dim %d --gamma %f --lr %f --max_step %d --log_interval %d \
--num_thread %d --batch_size_eval %d --neg_sample_size_eval %d \
--regularization_coef %f --num_client %d''' % (args.model_name, args.dataset, args.data_path, args.format,
                                            args.ip_config, args.hidden_dim, args.gamma, args.lr,
                                            total_client, args.model_name, args.dataset, args.data_path,
                                            args.format, args.save_path, args.ip_config, args.batch_size,
                                            args.neg_sample_size, args.hidden_dim, args.gamma, args.lr,
                                            args.max_step, args.log_interval, args.num_thread, args.batch_size_eval,
                                            args.neg_sample_size_eval, args.regularization_coef, args.num_client_proc)
    if args.test == True:
        cmd_str += ' --test'
    if args.no_save_emb == True:
        cmd_str += ' --no_save_emb'
    if args.neg_adversarial_sampling == True:
        cmd_str += ' -adv'

    file_path = os.path.join(args.path, SCRIPT_FILE)
    if os.path.exists(file_path):
        os.remove(file_path)

    f = open(file_path, 'w')
    f.write(cmd_str)
    f.close()

    st = os.stat(file_path)
    os.chmod(file_path, st.st_mode | stat.S_IEXEC)

    return file_path

def launch(args):
    """launch kvclient and kvserver processes to cluster
    """
    job_list = []
    cmd_list = []
    file_path = construct_cmd_script(args)
    # copy script file to remote machine
    print("Copy script to remote machine ...")
    with open(args.ip_config) as f:
        for line in f:
            ip, _, _ = line.strip().split(' ')
            if is_local(ip) == False:
                cmd_str = scp_file(file_path, ip, args.path, args.ssh_key)
                job_list.append(run_cmd(cmd_str))
                cmd_list.append(cmd_str)
    for i in range(len(job_list)):
        wait_job(job_list[i], cmd_list[i])
    # ssh and execute script
    job_list = []
    cmd_list = []
    with open(args.ip_config) as f:
        machine_id = 0
        for line in f:
            ip, _, count = line.strip().split(' ')
            id_low = machine_id * int(count)
            id_high = (machine_id+1) * int(count)
            cmd_str = 'cd %s; ./%s %d %d' % (args.path, SCRIPT_FILE, id_low, id_high)
            if is_local(ip) == False: # remote command
                cmd_str = ssh_cmd(cmd_str, ip, args.ssh_key)
                print(cmd_str)
            job_list.append(run_cmd(cmd_str))
            cmd_list.append(cmd_str)
            machine_id += 1
    # wait job finish
    for i in range(len(job_list)):
        wait_job(job_list[i], cmd_list[i])

def main():
    args = ArgParser().parse_args()
    launch(args)

if __name__ == '__main__':
    main()