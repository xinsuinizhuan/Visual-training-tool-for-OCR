import argparse
import time
import torch

from models import build_rec_model
from datasets import build_rec_dataloader
from losses import build_rec_loss
from metrics import build_rec_metric
from utils import character, load_rec_model, save_rec_model, CTCLabelConverter
from optimizer import build_optimizer, build_scheduler


def test_model(model, device, data_loader, converter, metric, loss_func, show_str_size):
    model.eval()
    with torch.no_grad():
        running_loss, running_char_corrects, running_all, running_all_char_correct, \
            running_word_corrects = 0., 0., 0., 0., 0.
        word_correct, char_correct = 0, 0
        batch_idx = 0
        show_strs = []
        for batch_idx, batch_data in enumerate(data_loader):
            targets, targets_lengths = converter.encode(batch_data['label'])
            batch_data['targets'] = targets
            batch_data['targets_lengths'] = targets_lengths
            batch_data['image'] = batch_data['image'].to(device)
            batch_data['targets'] = batch_data['targets'].to(device)
            batch_data['targets_lengths'] = batch_data['targets_lengths'].to(
                device)
            predicted = model.forward(
                batch_data['image'])
            loss_dict = loss_func(
                predicted, batch_data['targets'], batch_data['targets_lengths'])
            running_loss += loss_dict['loss'].item()
            acc_dict = metric(predicted, batch_data['label'])
            word_correct = acc_dict['word_correct']
            char_correct = acc_dict['char_correct']
            show_strs.extend(acc_dict['show_str'])
            running_char_corrects += char_correct
            running_word_corrects += word_correct
            running_all_char_correct += torch.sum(targets_lengths).item()
            running_all += len(batch_data['image'])
            if batch_idx == 0:
                since = time.time()
            elif batch_idx == len(data_loader)-1:
                print('Eval:[{:5.0f}/{:5.0f} ({:.0f}%)] loss:{:.4f} word acc:{:.4f} char acc:{:.4f} cost time:{:5.0f}s'.format(
                    running_all,
                    len(data_loader.dataset),
                    100. * batch_idx / (len(data_loader)-1),
                    running_loss / running_all,
                    running_word_corrects / running_all,
                    running_char_corrects / running_all_char_correct,
                    time.time()-since))

    for s in show_strs[:show_str_size]:
        print(s)
    model.train()
    return running_word_corrects / running_all, running_char_corrects / running_all_char_correct,


def train_model(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = build_rec_dataloader(
        cfg.train_root, cfg.train_list, cfg.batch_size, cfg.workers, character, is_train=True)
    test_loader = build_rec_dataloader(
        cfg.test_root, cfg.test_list, cfg.batch_size, cfg.workers, character, is_train=False)
    converter = CTCLabelConverter(character)
    loss_func = build_rec_loss().to(device)
    metric = build_rec_metric(converter)
    model = build_rec_model(cfg, converter.num_of_classes).to(device)
    if cfg.model_path != '':
        load_rec_model(cfg.model_path, model)
    optimizer = build_optimizer(model, cfg.lr)
    scheduler = build_scheduler(optimizer)
    val_word_accu, val_char_accu, best_word_accu = 0., 0., 0.
    for epoch in range(cfg.epochs):
        model.train()
        running_loss, running_char_corrects, running_all, running_all_char_correct, \
             running_word_corrects = 0., 0., 0., 0., 0.
        word_correct, char_correct = 0, 0
        for batch_idx, batch_data in enumerate(train_loader):
            batch_data['targets'], batch_data['targets_lengths'] = converter.encode(
                batch_data['label'])
            for key, value in batch_data.items():
                if value is not None:
                    if isinstance(value, torch.Tensor):
                        batch_data[key] = value.to(device)
            predicted = model.forward(
                batch_data['image'])
            loss_dict = loss_func(
                predicted, batch_data['targets'], batch_data['targets_lengths'])
            optimizer.zero_grad()
            loss_dict['loss'].backward()
            optimizer.step()
            running_loss += loss_dict['loss'].item()
            acc_dict = metric(predicted, batch_data['label'])
            word_correct = acc_dict['word_correct']
            char_correct = acc_dict['char_correct']
            running_char_corrects += char_correct
            running_word_corrects += word_correct
            running_all_char_correct += torch.sum(
                batch_data['targets_lengths']).item()
            running_all += len(batch_data['image'])

            if batch_idx == 0:
                since = time.time()
            elif batch_idx % cfg.display_interval == 0 or (batch_idx == len(train_loader)-1):
                print('Train:[epoch {}/{} {:5.0f}/{:5.0f} ({:.0f}%)] loss:{:.4f} word acc:{:.4f} char acc:{:.4f} cost time:{:5.0f}s estimated time:{:5.0f}s'.format(
                    epoch+1,
                    cfg.epochs,
                    running_all,
                    len(train_loader.dataset),
                    100. * batch_idx / (len(train_loader)-1),
                    running_loss / running_all,
                    running_word_corrects/running_all,
                    running_char_corrects / running_all_char_correct,
                    time.time()-since,
                    (time.time()-since)*(len(train_loader)-1) / batch_idx - (time.time()-since)))
            if batch_idx != 0 and batch_idx % cfg.val_interval == 0:
                val_word_accu, val_char_accu = test_model(
                    model, device, test_loader, converter, metric, loss_func, cfg.show_str_size)
                if val_word_accu > best_word_accu:
                    best_word_accu = val_word_accu
                    save_rec_model(cfg.model_type, model,  'best',
                                   best_word_accu, val_char_accu)
        if epoch % cfg.save_epoch == 0:
            val_word_accu, val_char_accu = test_model(
                model, device, test_loader, converter, metric, loss_func, cfg.show_str_size)
            save_rec_model(cfg.model_type, model, epoch,
                           val_word_accu, val_char_accu)
        scheduler.step()


def main():
    parser = argparse.ArgumentParser(description='rec net')
    parser.add_argument('--train_root', default='E:/precode/',
                        help='path to train dir')
    parser.add_argument('--test_root', default='E:/precode/',
                        help='path to test dir')
    parser.add_argument(
        '--train_list', default='E:/precode/train1.txt', help='path to train label')
    parser.add_argument(
        '--test_list', default='E:/precode/test1.txt', help='path to test label')
    parser.add_argument('--model_path', default='',
                        help='path to model')
    parser.add_argument('--model_type', default='crnn',
                        help='model type', type=str)
    parser.add_argument('--use_lstm', default=False, action='store_true',
                        help='use lstm')
    parser.add_argument('--nh', default=256, help='nh', type=int)
    parser.add_argument('--lr', default=0.0001,
                        help='initial learning rate', type=float)
    parser.add_argument('--batch_size', default=8, type=int,
                        help='mini-batch size (default: 8)')
    parser.add_argument('--workers', default=0,
                        help='number of data loading workers (default: 0)', type=int)
    parser.add_argument('--epochs', default=50,
                        help='number of total epochs', type=int)
    parser.add_argument('--display_interval', default=20,
                        help='display interval', type=int)
    parser.add_argument('--val_interval', default=200,
                        help='val interval', type=int)
    parser.add_argument('--save_epoch', default=1,
                        help='save epoch', type=int)
    parser.add_argument('--show_str_size', default=10,
                        help='show str size', type=int)
    cfg = parser.parse_args()
    train_model(cfg)


if __name__ == '__main__':
    main()
