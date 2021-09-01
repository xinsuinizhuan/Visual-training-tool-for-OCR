import os
import torch


def save_rec_model(model_type, model,  epoch, word_acc,char_acc):
    if epoch == 'best':
        save_path='./save_model/{}_best_rec.pth'.format(model_type)
        if os.path.exists(save_path):
            data = torch.load(save_path)
            if 'model' in data and data['word_acc'] > word_acc:
                return
        torch.save({
            'model': model.state_dict(),
            'word_acc': word_acc,
            'char_acc': char_acc},
            save_path)
    else:
        save_path='./save_model/{}_epoch{}_word_acc{:05f}_char_acc{:05f}.pth'.format(model_type,epoch,word_acc,char_acc)
        torch.save({
            'model': model.state_dict(),
            'word_acc': word_acc,
            'char_acc': char_acc},
            save_path)
    print('save model to:'+save_path)

def load_rec_model(model_path, model):
    data = torch.load(model_path)
    if 'model' in data:
        model.load_state_dict(data['model'])
        print('Model loaded word_acc {} ,char_acc {}'.format(data['word_acc'],data['char_acc']))
