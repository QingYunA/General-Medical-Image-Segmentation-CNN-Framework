
import os
import shutil
from sklearn.model_selection import train_test_split
from tqdm import tqdm

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='split train and test data')
    parser.add_argument('--data_path', type=str, default='', help='the source path of your data')
    parser.add_argument('-t', '--target_path', type=str, default='', help='the target path of test')
    parser.add_argument('--test_size', type=float, default=0.5, help='the size of test data')
    args = parser.parse_args()

    data_path = args.data_path
    target_path = args.target_path

    target_train = os.path.join(target_path, str(args.test_size * 100) + '_train')
    target_test = os.path.join(target_path, str(args.test_size * 100) + '_test')
    os.makedirs(target_train, exist_ok=True)
    os.makedirs(target_test, exist_ok=True)

    hgg_ls = os.listdir(os.path.join(data_path, 'HGG'))
    lgg_ls = os.listdir(os.path.join(data_path, 'LGG'))

    hgg_train, hgg_test = train_test_split(hgg_ls, test_size=args.test_size)
    lgg_train, lgg_test = train_test_split(lgg_ls, test_size=args.test_size)

    train_ls = hgg_train + lgg_train
    test_ls = hgg_test + lgg_test

    #* copy train and test data to target dir
    for i in tqdm(train_ls, total=len(train_ls), desc='copy train data'):
        try:
            shutil.copytree(os.path.join(data_path, 'HGG', i), os.path.join(target_train, i))
        except:
            shutil.copytree(os.path.join(data_path, 'LGG', i), os.path.join(target_train, i))

    for i in tqdm(test_ls, total=len(test_ls), desc='copy test data'):
        try:
            shutil.copytree(os.path.join(data_path, 'HGG', i), os.path.join(target_test, i))
        except:
            shutil.copytree(os.path.join(data_path, 'LGG', i), os.path.join(target_test, i))

    print('sucessful copy train and test data to target dir!')
    print('train length:{}'.format(len(train_ls)))
    print('test length:{}'.format(len(test_ls)))