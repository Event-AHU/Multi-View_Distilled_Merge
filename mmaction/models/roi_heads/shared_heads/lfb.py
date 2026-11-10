
import io
import os
import os.path as osp
import warnings

import torch
import torch.distributed as dist
from mmengine.dist import get_dist_info

try:
    import lmdb
    lmdb_imported = True
except (ImportError, ModuleNotFoundError):
    lmdb_imported = False


class LFB:
    

    def __init__(self,
                 lfb_prefix_path,
                 max_num_sampled_feat=5,
                 window_size=60,
                 lfb_channels=2048,
                 dataset_modes=('train', 'val'),
                 device='gpu',
                 lmdb_map_size=4e9,
                 construct_lmdb=True):
        if not osp.exists(lfb_prefix_path):
            raise ValueError(
                f'lfb prefix path {lfb_prefix_path} does not exist!')
        self.lfb_prefix_path = lfb_prefix_path
        self.max_num_sampled_feat = max_num_sampled_feat
        self.window_size = window_size
        self.lfb_channels = lfb_channels
        if not isinstance(dataset_modes, tuple):
            assert isinstance(dataset_modes, str)
            dataset_modes = (dataset_modes, )
        self.dataset_modes = dataset_modes
        self.device = device

        rank, world_size = get_dist_info()

        if self.device == 'gpu':
            if 'LOCAL_RANK' in os.environ:
                local_rank = int(os.environ['LOCAL_RANK'])
            else:
                gpus_per_node = torch.cuda.device_count()
                local_rank = rank % gpus_per_node

            self.load_lfb(f'cuda:{local_rank}')
        elif self.device == 'cpu':
            if world_size > 1:
                warnings.warn(
                    'If distributed training is used with multi-GPUs, lfb '
                    'will be loaded multiple times on RAM. In this case, '
                    "'lmdb' is recommended.", UserWarning)
            self.load_lfb('cpu')
        elif self.device == 'lmdb':
            assert lmdb_imported, (
                'Please install `lmdb` to load lfb on lmdb!')
            self.lmdb_map_size = lmdb_map_size
            self.construct_lmdb = construct_lmdb
            self.lfb_lmdb_path = osp.normpath(
                osp.join(self.lfb_prefix_path, 'lmdb'))

            if rank == 0 and self.construct_lmdb:
                print('Constructing LFB lmdb...')
                self.load_lfb_on_lmdb()


            if world_size > 1:
                dist.barrier()
            self.lmdb_env = lmdb.open(self.lfb_lmdb_path, readonly=True)
        else:
            raise ValueError("Device must be 'gpu', 'cpu' or 'lmdb', ",
                             f'but get {self.device}.')

    def load_lfb(self, map_location):
        self.lfb = {}
        for dataset_mode in self.dataset_modes:
            lfb_path = osp.normpath(
                osp.join(self.lfb_prefix_path, f'lfb_{dataset_mode}.pkl'))
            print(f'Loading LFB from {lfb_path}...')
            self.lfb.update(torch.load(lfb_path, map_location=map_location))

        for video_id in self.lfb:
            video_features = self.lfb[video_id]
            for sec in video_features:
                if isinstance(video_features[sec], (list, tuple)):
                    video_features[sec] = torch.stack(video_features[sec])
            self.lfb[video_id] = video_features
        print(f'LFB has been loaded on {map_location}.')

    def load_lfb_on_lmdb(self):
        lfb = {}
        for dataset_mode in self.dataset_modes:
            lfb_path = osp.normpath(
                osp.join(self.lfb_prefix_path, f'lfb_{dataset_mode}.pkl'))
            lfb.update(torch.load(lfb_path, map_location='cpu'))

        lmdb_env = lmdb.open(self.lfb_lmdb_path, map_size=self.lmdb_map_size)
        for key, value in lfb.items():
            txn = lmdb_env.begin(write=True)
            buff = io.BytesIO()
            torch.save(value, buff)
            buff.seek(0)
            txn.put(key.encode(), buff.read())
            txn.commit()
            buff.close()

        print(f'LFB lmdb has been constructed on {self.lfb_lmdb_path}!')

    def sample_long_term_features(self, video_id, timestamp):
        if self.device == 'lmdb':
            with self.lmdb_env.begin(write=False) as txn:
                buf = txn.get(video_id.encode())
                video_features = torch.load(io.BytesIO(buf))
        else:
            video_features = self.lfb[video_id]


        window_size, K = self.window_size, self.max_num_sampled_feat
        start = timestamp - (window_size // 2)
        lt_feats = torch.zeros(window_size, K, self.lfb_channels)

        for idx, sec in enumerate(range(start, start + window_size)):
            if sec in video_features:

                feat = video_features[sec]
                num_feat = feat.shape[0]


                random_lfb_indices = torch.randperm(num_feat)[:K]
                lt_feats[idx, :num_feat] = feat[random_lfb_indices]


        return lt_feats.reshape(-1, self.lfb_channels)

    def __getitem__(self, img_key):
        
        video_id, timestamp = img_key.split(',')
        return self.sample_long_term_features(video_id, int(timestamp))

    def __len__(self):
        
        return len(self.lfb)
