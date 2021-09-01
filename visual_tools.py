import os
import threading
import subprocess
from exduir import *
from ctypes import *


@ctypes.PYFUNCTYPE(c_longlong, c_size_t, c_int, c_int, c_size_t, c_size_t, c_void_p)
def msg_proc(hwnd: int, handle: int, umsg: int, wparam: int, lparam: int, lpresult: int) -> int:
    if umsg==16:#WM_CLOSE
        print('窗口关闭')
    return 0


@ctypes.PYFUNCTYPE(c_longlong, c_int, c_int, c_int, c_size_t, c_size_t)
def on_button_event(hobj: int, id: int, code: int, wparam: int, lparam: int) -> int:
    if code == NM_CLICK:
        if hobj == hobj_button_train:
            train_path = ex.objGetText(hobj_edit_train_path)
            train_label = ex.objGetText(hobj_edit_train_label)
            test_path = ex.objGetText(hobj_edit_test_path)
            test_label = ex.objGetText(hobj_edit_test_label)
            model_path = ex.objGetText(hobj_edit_model_path)
            for hobj_radio in hobj_radiolist:
                if ex.objSendMessage(hobj_radio, BM_GETCHECK, 0, 0) == 1:
                    model_type = ex.objGetText(hobj_radio)
            lr = ex.objGetText(hobj_edit_lr)
            batchsize = ex.objGetText(hobj_edit_batchsize)
            epochs = ex.objGetText(hobj_edit_epochs)
            display_interval = ex.objGetText(hobj_edit_display_interval)
            val_interval = ex.objGetText(hobj_edit_val_interval)
            save_epoch = ex.objGetText(hobj_edit_save_epoch)
            show_str_size = ex.objGetText(hobj_edit_show_str_size)
            nh = ex.objGetText(hobj_edit_nh)
            use_lstm = ex.objSendMessage(hobj_switch_lstm, BM_GETCHECK, 0, 0)

            command_line = ('python rec_train.py',
                            (' --train_root ' +
                             train_path) if os.path.exists(train_path) else '',
                            (' --test_root ' +
                             test_path) if os.path.exists(test_path) else '',
                            (' --train_list ' +
                             train_label) if os.path.exists(train_label) else '',
                            (' --test_list ' +
                             test_label) if os.path.exists(test_label) else '',
                            (' --model_path ' +
                             model_path) if os.path.exists(model_path) else '',
                            ' --model_type ' + model_type,
                            ' --use_lstm ' if use_lstm == 1 else '',
                            ' --nh ' + nh,
                            ' --lr ' + lr,
                            ' --batch_size ' + batchsize,
                            ' --epochs ' + epochs,
                            ' --display_interval ' + display_interval,
                            ' --val_interval ' + val_interval,
                            ' --save_epoch ' + save_epoch,
                            ' --show_str_size ' + show_str_size,
                            )
            command_line = ''.join(command_line)
            thread1 = threading.Thread(target=train, args=(command_line,))
            thread1.start()
            ex.objSetFocus(hobj_edit_print)
            ex.objEnable(hobj_button_train, False)
            ex.objEnable(hobj_button_stop, True)
        elif hobj == hobj_button_stop:
            process_control.stop()
            ex.objEnable(hobj_button_stop, False)
            ex.objEnable(hobj_button_train, True)
    return 0


class ProcessControl:
    def __init__(self) -> None:
        pass

    def train(self, command_line):
        self.process = subprocess.Popen(
            command_line, shell=True, stdout=subprocess.PIPE, encoding="utf-8")
        while self.process.poll() == None:
            outline = self.process.stdout.readline()
            ex.objSendMessage(hobj_edit_print, EM_SETSEL, -1, -1)
            ex.objSendMessage(hobj_edit_print, EM_REPLACESEL, -1, outline)
        ex.objEnable(hobj_button_stop, False)
        ex.objEnable(hobj_button_train, True)

    def stop(self):
        self.process.terminate()


process_control = ProcessControl()


def train(command_line):
    process_control.train(command_line)


if __name__ == '__main__':

    ex = ExDUIR()
    hwnd = ex.wndCreate(0, '可视化训练工具', 0, 0, 800, 800)
    hexdui = ex.duiBindWindowEx(hwnd, EWS_MAINWINDOW | EWS_BUTTON_CLOSE | EWS_BUTTON_MIN |
                                EWS_MOVEABLE | EWS_CENTERWINDOW | EWS_TITLE | EWS_HASICON, 0, msg_proc)
    ex.duiSetLong(hexdui, EWL_CRBKG, ex.RGBA(140, 140, 140, 255))
    hobj_label_train_path = ex.objCreateEx(-1, 'static', '训练数据路径:', -1,
                                           10, 40, 100, 30, hexdui, 0, DT_RIGHT | DT_VCENTER, 0, 0)
    hobj_edit_train_path = ex.objCreate(
        'edit', 'E:/precode/', -1, 120, 40, 670, 30, hexdui)

    hobj_label_train_label = ex.objCreateEx(
        -1, 'static', '训练数据标签:', -1, 10, 80, 100, 30, hexdui, 0, DT_RIGHT | DT_VCENTER, 0, 0)
    hobj_edit_train_label = ex.objCreate(
        'edit', 'E:/precode/train1.txt', -1, 120, 80, 670, 30, hexdui)

    hobj_label_test_path = ex.objCreateEx(-1,
                                          'static', '测试数据路径:', -1, 10, 120, 100, 30, hexdui, 0, DT_RIGHT | DT_VCENTER, 0, 0)
    hobj_edit_test_path = ex.objCreate(
        'edit', 'E:/precode/', -1, 120, 120, 670, 30, hexdui)

    hobj_label_test_label = ex.objCreateEx(-1,
                                           'static', '测试数据标签:', -1, 10, 160, 100, 30, hexdui, 0, DT_RIGHT | DT_VCENTER, 0, 0)
    hobj_edit_test_label = ex.objCreate(
        'edit', 'E:/precode/test1.txt', -1, 120, 160, 670, 30, hexdui)

    hobj_label_model_path = ex.objCreateEx(-1,
                                           'static', '预训练模型路径:', -1, 10, 200, 100, 30, hexdui, 0, DT_RIGHT | DT_VCENTER, 0, 0)
    hobj_edit_model_path = ex.objCreate(
        'edit', 'save_model/crnn_best_rec.pth', -1, 120, 200, 670, 30, hexdui)

    hobj_label_model_type = ex.objCreateEx(-1,
                                           'static', '模型类型:', -1, 10, 240, 100, 30, hexdui, 0, DT_RIGHT | DT_VCENTER, 0, 0)
    hobj_radiolist = []
    hobj_radiolist.append(ex.objCreate(
        'radiobutton', 'crnn', -1, 120, 240, 50, 30, hexdui))
    hobj_radiolist.append(ex.objCreate(
        'radiobutton', 'resnet18', -1, 170, 240, 70, 30, hexdui))
    hobj_radiolist.append(ex.objCreate(
        'radiobutton', 'resnet34', -1, 240, 240, 70, 30, hexdui))
    hobj_radiolist.append(ex.objCreate(
        'radiobutton', 'resnet50', -1, 310, 240, 70, 30, hexdui))
    hobj_radiolist.append(ex.objCreate(
        'radiobutton', 'resnet101', -1, 380, 240, 80, 30, hexdui))
    hobj_radiolist.append(ex.objCreate(
        'radiobutton', 'resnet152', -1, 460, 240, 80, 30, hexdui))
    hobj_radiolist.append(ex.objCreate(
        'radiobutton', 'resnet200', -1, 540, 240, 80, 30, hexdui))
    hobj_radiolist.append(ex.objCreate(
        'radiobutton', 'mbv3_large', -1, 620, 240, 90, 30, hexdui))
    hobj_radiolist.append(ex.objCreate(
        'radiobutton', 'mbv3_small', -1, 710, 240, 90, 30, hexdui))
    ex.objSendMessage(hobj_radiolist[0], BM_SETCHECK, 1, 0)

    hobj_label_lr = ex.objCreateEx(-1,
                                   'static', '学习率lr:', -1, 10, 280, 100, 30, hexdui, 0, DT_RIGHT | DT_VCENTER, 0, 0)
    hobj_edit_lr = ex.objCreate(
        'edit', '0.0001', -1, 120, 280, 100, 30, hexdui)

    hobj_label_batchsize = ex.objCreateEx(-1,
                                          'static', '批大小batch_size:', -1, 320, 280, 100, 30, hexdui, 0, DT_RIGHT | DT_VCENTER, 0, 0)
    hobj_edit_batchsize = ex.objCreate(
        'edit', '8', -1, 430, 280, 100, 30, hexdui)

    hobj_label_epochs = ex.objCreateEx(-1,
                                       'static', '训练批次epochs:', -1, 580, 280, 100, 30, hexdui, 0, DT_RIGHT | DT_VCENTER, 0, 0)
    hobj_edit_epochs = ex.objCreate('edit', '5', -1, 690, 280, 100, 30, hexdui)

    hobj_label_display_interval = ex.objCreateEx(-1,
                                                 'static', '显示步长:', -1, 10, 320, 100, 30, hexdui, 0, DT_RIGHT | DT_VCENTER, 0, 0)
    hobj_edit_display_interval = ex.objCreate(
        'edit', '20', -1, 120, 320, 100, 30, hexdui)

    hobj_label_val_interval = ex.objCreateEx(-1,
                                             'static', '验证步长:', -1, 320, 320, 100, 30, hexdui, 0, DT_RIGHT | DT_VCENTER, 0, 0)
    hobj_edit_val_interval = ex.objCreate(
        'edit', '200', -1, 430, 320, 100, 30, hexdui)

    hobj_label_save_epoch = ex.objCreateEx(-1,
                                           'static', '保存间隔批次:', -1, 580, 320, 100, 30, hexdui, 0, DT_RIGHT | DT_VCENTER, 0, 0)
    hobj_edit_save_epoch = ex.objCreate(
        'edit', '1', -1, 690, 320, 100, 30, hexdui)

    hobj_label_nh = ex.objCreateEx(-1,
                                   'static', '特征宽度nh:', -1, 10, 360, 100, 30, hexdui, 0, DT_RIGHT | DT_VCENTER, 0, 0)
    hobj_edit_nh = ex.objCreate('edit', '256', -1, 120, 360, 100, 30, hexdui)

    hobj_label_show_str_size = ex.objCreateEx(-1,
                                              'static', '验证文本数量:', -1, 320, 360, 100, 30, hexdui, 0, DT_RIGHT | DT_VCENTER, 0, 0)
    hobj_edit_show_str_size = ex.objCreate(
        'edit', '10', -1, 430, 360, 100, 30, hexdui)

    hobj_label_lstm = ex.objCreateEx(-1,
                                     'static', '是否采用lstm:', -1, 580, 360, 100, 30, hexdui, 0, DT_RIGHT | DT_VCENTER, 0, 0)
    hobj_switch_lstm = ex.objCreate(
        'switch', '是|否', -1, 690, 360, 100, 30, hexdui)

    hobj_button_train = ex.objCreate(
        'button', '开始训练', -1, 10, 400, 100, 30, hexdui)
    ex.objHandleEvent(hobj_button_train, NM_CLICK, on_button_event)

    hobj_button_stop = ex.objCreate(
        'button', '停止训练', -1, 120, 400, 100, 30, hexdui)
    ex.objHandleEvent(hobj_button_stop, NM_CLICK, on_button_event)
    ex.objEnable(hobj_button_stop, False)
    hobj_edit_print = ex.objCreateEx(EOS_EX_FOCUSABLE | EOS_EX_COMPOSITED, 'edit',
                                     '', EOS_VISIBLE | EOS_VSCROLL, 10, 440, 780, 350, hexdui, 0, DT_VCENTER, 0, 0)

    ex.duiShowWindow(hexdui)
    ex.wndMsgLoop()
    ex.unInit()
