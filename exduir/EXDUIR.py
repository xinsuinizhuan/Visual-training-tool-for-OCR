import ctypes
from ctypes import *
import os

DIR = os.path.dirname(os.path.abspath(__file__))
DLL_PATH = os.path.join(DIR, 'libexdui.dll')
EXT_PATH = os.path.join(DIR, 'Default.ext')

libexdui = ctypes.CDLL(DLL_PATH)


BIF_DEFAULT = 0
BIF_DISABLESCALE = 2
BIF_GRID_EXCLUSION_CENTER = 4
BIF_PLAYIMAGE = 1
BIF_POSITION_X_PERCENT = 16
BIF_POSITION_Y_PERCENT = 8
BIR_DEFAULT = 0
BIR_NO_REPEAT = 1
BIR_REPEAT = 2
BIR_REPEAT_X = 3
BIR_REPEAT_Y = 4
BM_GETCHECK = 0x00F0
BM_SETCHECK = 0x00F1
CBN_CLOSEUP = 8
CBN_DROPDOWN = 7
CBN_EDITCHANGE = 5
CBN_POPUPLISTWINDOW = 2001
CBN_SELCHANGE = 1
CFM_BOLD = 0x00000001
CFM_COLOR = 0x40000000
CFM_FACE = 0x20000000
CFM_ITALIC = 0x00000002
CFM_LINK = 0x00000020
CFM_OFFSET = 0x10000000
CFM_SIZE = 0x80000000
CFM_STRIKEOUT = 0x00000008
CFM_UNDERLINE = 0x00000004
COLOR_EX_BACKGROUND = 0
COLOR_EX_BORDER = 1
COLOR_EX_EDIT_BANNER = 31
COLOR_EX_EDIT_CARET = 30
COLOR_EX_TEXT_CHECKED = 6
COLOR_EX_TEXT_DOWN = 4
COLOR_EX_TEXT_FOCUS = 5
COLOR_EX_TEXT_HOT = 8
COLOR_EX_TEXT_HOVER = 3
COLOR_EX_TEXT_NORMAL = 2
COLOR_EX_TEXT_SELECT = 7
COLOR_EX_TEXT_SHADOW = 10
COLOR_EX_TEXT_VISTED = 9
CVC_DX_D2DBITMAP = 2
CVC_DX_D2DCONTEXT = 1
CV_COMPOSITE_MODE_SRCCOPY = 1
CV_COMPOSITE_MODE_SRCOVER = 0
DT_TOP = 0x00000000
DT_LEFT = 0x00000000
DT_CENTER = 0x00000001
DT_RIGHT = 0x00000002
DT_VCENTER = 0x00000004
DT_BOTTOM = 0x00000008
DT_WORDBREAK = 0x00000010
DT_SINGLELINE = 0x00000020
DT_EXPANDTABS = 0x00000040
DT_TABSTOP = 0x00000080
DT_NOPREFIX = 0x00000800
EBS_CHECKBUTTON = 1
EBS_ICONRIGHT = 8
EBS_RADIOBUTTON = 2
EBS_TEXTOFFSET = 4
ECS_ALLOWEDIT = 1
ECVF_CANVASANTIALIAS = 0x01
ECVF_CLIPED = 0x80000000
ECVF_GDI_COMPATIBLE = 0x40000000
ECVF_TEXTANTIALIAS = 0x02
EES_ALLOWBEEP = 0x10
EES_ALLOWTAB = 0x800
EES_AUTOWORDSEL = 0x100
EES_DISABLEDRAG = 0x01
EES_DISABLEMENU = 0x200
EES_HIDDENCARET = 0x2000
EES_HIDESELECTION = 0x04
EES_NEWLINE = 0x40
EES_NUMERICINPUT = 0x80
EES_PARSEURL = 0x400
EES_PAUSE = 1
EES_PLAY = 0
EES_READONLY = 0x20
EES_RICHTEXT = 0x08
EES_SHOWTIPSALWAYS = 0x1000
EES_STOP = 2
EES_UNDERLINE = 0x4000
EES_USEPASSWORD = 0x02
EFS_BOLD = 1
EFS_DEFAULT = 0
EFS_ITALIC = 2
EFS_STRICKOUT = 8
EFS_UNDERLINE = 4
EILVS_BUTTON = 0x400
ELCP_ABSOLUTE_BOTTOM = 7
ELCP_ABSOLUTE_BOTTOM_TYPE = 8
ELCP_ABSOLUTE_HEIGHT = 11
ELCP_ABSOLUTE_HEIGHT_TYPE = 12
ELCP_ABSOLUTE_LEFT = 1
ELCP_ABSOLUTE_LEFT_TYPE = 2
ELCP_ABSOLUTE_OFFSET_H = 13
ELCP_ABSOLUTE_OFFSET_H_TYPE = 14
ELCP_ABSOLUTE_OFFSET_V = 15
ELCP_ABSOLUTE_OFFSET_V_TYPE = 16
ELCP_ABSOLUTE_RIGHT = 5
ELCP_ABSOLUTE_RIGHT_TYPE = 6
ELCP_ABSOLUTE_TOP = 3
ELCP_ABSOLUTE_TOP_TYPE = 4
ELCP_ABSOLUTE_TYPE_OBJPS = 3
ELCP_ABSOLUTE_TYPE_PS = 2
ELCP_ABSOLUTE_TYPE_PX = 1
ELCP_ABSOLUTE_TYPE_UNKNOWN = 0
ELCP_ABSOLUTE_WIDTH = 9
ELCP_ABSOLUTE_WIDTH_TYPE = 10
ELCP_FLOW_NEW_LINE = 2
ELCP_FLOW_SIZE = 1
ELCP_LINEAR_ALGIN_FILL = 0
ELCP_LINEAR_ALIGN = 2
ELCP_LINEAR_ALIGN_CENTER = 2
ELCP_LINEAR_ALIGN_LEFT_TOP = 1
ELCP_LINEAR_ALIGN_RIGHT_BOTTOM = 3
ELCP_LINEAR_SIZE = 1
ELCP_MARGIN_BOTTOM = -4
ELCP_MARGIN_LEFT = -1
ELCP_MARGIN_RIGHT = -3
ELCP_MARGIN_TOP = -2
ELCP_PAGE_FILL = 1
ELCP_RELATIVE_BOTTOM_ALIGN_OF = 8
ELCP_RELATIVE_BOTTOM_OF = 4
ELCP_RELATIVE_CENTER_PARENT_H = 9
ELCP_RELATIVE_CENTER_PARENT_V = 10
ELCP_RELATIVE_LEFT_ALIGN_OF = 5
ELCP_RELATIVE_LEFT_OF = 1
ELCP_RELATIVE_RIGHT_ALIGN_OF = 7
ELCP_RELATIVE_RIGHT_OF = 3
ELCP_RELATIVE_TOP_ALIGN_OF = 6
ELCP_RELATIVE_TOP_OF = 2
ELCP_TABLE_CELL = 2
ELCP_TABLE_CELL_SPAN = 4
ELCP_TABLE_FILL = 5
ELCP_TABLE_ROW = 1
ELCP_TABLE_ROW_SPAN = 3
ELDS_LINE = 0x01
ELN_CHECKCHILDPROPVALUE = 8
ELN_CHECKPROPVALUE = 7
ELN_FILL_XML_CHILD_PROPS = 10
ELN_FILL_XML_PROPS = 9
ELN_GETCHILDPROPCOUNT = 2
ELN_GETPROPSCOUNT = 1
ELN_INITCHILDPROPS = 5
ELN_INITPROPS = 3
ELN_UNINITCHILDPROPS = 6
ELN_UNINITPROPS = 4
ELN_UPDATE = 15
ELP_DIRECTION_H = 0
ELP_DIRECTION_V = 1
ELP_FLOW_DIRECTION = 1
ELP_LINEAR_DALIGN = 2
ELP_LINEAR_DALIGN_CENTER = 1
ELP_LINEAR_DALIGN_LEFT_TOP = 0
ELP_LINEAR_DALIGN_RIGHT_BOTTOM = 2
ELP_LINEAR_DIRECTION = 1
ELP_PADDING_BOTTOM = -4
ELP_PADDING_LEFT = -1
ELP_PADDING_RIGHT = -3
ELP_PADDING_TOP = -2
ELP_PAGE_CURRENT = 1
ELP_TABLE_ARRAY_CELL = 2
ELP_TABLE_ARRAY_ROW = 1
ELT_ABSOLUTE = 6
ELT_FLOW = 2
ELT_LINEAR = 1
ELT_NULL = 0
ELT_PAGE = 3
ELT_RELATIVE = 5
ELT_TABLE = 4
ELVS_ALLOWMULTIPLE = 0x08
ELVS_HORIZONTALLIST = 0x01
ELVS_ITEMTRACKING = 0x10
ELVS_SHOWSELALWAYS = 0x20
ELVS_VERTICALLIST = 0x00
EM_REPLACESEL = 0x00C2
EM_SETSEL = 0x00B1
EMBF_CENTEWINDOW = 0x40000000
EMBF_SHOWTIMEOUT = 0x20000000
EMBF_WINDOWICON = 0x80000000
EMNF_NOSHADOW = 0x80000000
EM_EXSETSEL = 1079
EM_FINDTEXTW = 1147
EM_GETTEXTRANGE = 1099
EM_LOAD_RTF = 6001
EM_REDO = 1108
EM_SETCUEBANNER = 5377
EM_SETTEXTEX = 1121
EM_UNDO = 199
EN_LINK = 1803
EN_SELCHANGE = 1794
EOL_ALPHA = -5
EOL_BLUR = -2
EOL_CURSOR = -17
EOL_EXSTYLE = -20
EOL_HCANVAS = -22
EOL_HFONT = -19
EOL_ID = -12
EOL_LPARAM = -7
EOL_LPWZTITLE = -28
EOL_NODEID = -1
EOL_OBJPARENT = -8
EOL_OBJPROC = -4
EOL_OWNER = -23
EOL_STATE = -24
EOL_STYLE = -16
EOL_TEXTFORMAT = -11
EOL_USERDATA = -21
EOP_DEFAULT = 0x80000000
EOS_BORDER = 0x20000000
EOS_DISABLED = 0x8000000
EOS_DISABLENOSCROLL = 0x2000000
EOS_EX_ACCEPTFILES = 0x4000000
EOS_EX_AUTOSIZE = 0x400000
EOS_EX_BLUR = 0x1000000
EOS_EX_COMPOSITED = 0x40000000
EOS_EX_CUSTOMDRAW = 0x80000000
EOS_EX_DRAGDROP = 0x2000000
EOS_EX_FOCUSABLE = 0x8000000
EOS_EX_TABSTOP = 0x10000000
EOS_EX_TOPMOST = 0x20000000
EOS_EX_TRANSPARENT = 0x800000
EOS_HSCROLL = 0x80000000
EOS_SIZEBOX = 0x4000000
EOS_VISIBLE = 0x10000000
EOS_VSCROLL = 0x40000000
EPDF_FILES = 255
EPDF_THEME = 254
EPF_DISABLESCALE = 1
EPP_BEGIN = 0
EPP_BKG = 1
EPP_BORDER = 2
EPP_CUSTOMDRAW = 3
EPP_END = 4
ERIBS_ROTATE = 0x01
ERLS_DRAWHORIZONTALLINE = 0x100
ERLS_DRAWVERTICALLINE = 0x200
ERLS_NOHEAD = 0x400
ERLV_CS_CLICKABLE = 0x01
ERLV_CS_LOCKWIDTH = 0x02
ERLV_CS_SORTABLE = 0x04
ERLV_RS_CHECKBOX = 0x01
ERLV_RS_CHECKBOX_CHECK = 0x02
ESBS_HORIZONTAL = 0x00
ESBS_VERTICAL = 0x01
ESS_CONTROLBUTTON = 8
ESS_HORIZONTALSCROLL = 0
ESS_LEFTTOPALIGN = 2
ESS_RIGHTBOTTOMALIGN = 4
ESS_VERTICALSCROLL = 1
ES_BACKANDFORTH = 0x20
ES_CALLFUNCTION = 0x40
ES_CYCLE = 0x02
ES_DISPATCHNOTIFY = 0x80
ES_MANYTIMES = 0x04
ES_ORDER = 0x08
ES_RELEASECURVE = 0x200
ES_REVERSE = 0x10
ES_SINGLE = 0x01
ES_THREAD = 0x100
ETS_SHOWADDANDSUB = 0x40
ETS_SHOWCABLE = 0x80
ET_CURVE = 51
ET_CUSTOM = 50
ET_Clerp = 2
ET_InBack = 29
ET_InBounce = 26
ET_InCirc = 23
ET_InCubic = 8
ET_InElastic = 32
ET_InExpo = 20
ET_InOutBack = 31
ET_InOutBounce = 28
ET_InOutCirc = 25
ET_InOutCubic = 10
ET_InOutElastic = 34
ET_InOutExpo = 22
ET_InOutQuad = 7
ET_InOutQuart = 13
ET_InOutQuint = 16
ET_InOutSine = 19
ET_InQuad = 5
ET_InQuart = 11
ET_InQuint = 14
ET_InSine = 17
ET_Linear = 1
ET_OutBack = 30
ET_OutBounce = 27
ET_OutCirc = 24
ET_OutCubic = 9
ET_OutElastic = 33
ET_OutExpo = 21
ET_OutQuad = 6
ET_OutQuart = 12
ET_OutQuint = 15
ET_OutSine = 18
ET_Punch = 4
ET_Spring = 3
EWL_ALPHA = -5
EWL_BLUR = -2
EWL_CRBKG = -31
EWL_CRBORDER = -30
EWL_HTHEME = 2
EWL_HWND = -6
EWL_LPARAM = -7
EWL_MINHEIGHT = -33
EWL_MINWIDTH = -34
EWL_MSGPROC = -4
EWL_OBJCAPTION = -54
EWL_OBJFOCUS = -53
EWS_BUTTON_CLOSE = 0x01
EWS_BUTTON_HELP = 0x40
EWS_BUTTON_MAX = 0x02
EWS_BUTTON_MENU = 0x08
EWS_BUTTON_MIN = 0x04
EWS_BUTTON_SETTING = 0x20
EWS_BUTTON_SKIN = 0x10
EWS_CENTERWINDOW = 0x20000
EWS_COMBOWINDOW = 0x100000
EWS_ESCEXIT = 0x8000
EWS_FULLSCREEN = 0x200
EWS_HASICON = 0x80
EWS_MAINWINDOW = 0x10000
EWS_MENU = 0x40000000
EWS_MESSAGEBOX = 0x80000000
EWS_MOVEABLE = 0x800
EWS_NOCAPTIONTOPMOST = 0x40000
EWS_NOINHERITBKG = 0x2000
EWS_NOSHADOW = 0x1000
EWS_NOTABBORDER = 0x4000
EWS_POPUPWINDOW = 0x80000
EWS_SIZEABLE = 0x400
EWS_TITLE = 0x100
EXGF_DPI_ENABLE = 0x02
EXGF_IMAGE_ANTIALIAS = 0x1000
EXGF_JS_ALL = 0x780000
EXGF_JS_FILE = 0x80000
EXGF_JS_MEMORY = 0x100000
EXGF_JS_MEMORY_ALLOC = 0x200000
EXGF_JS_PROCESS = 0x400000
EXGF_MENU_ALL = 0x800000
EXGF_OBJECT_DISABLEANIMATION = 0x10000
EXGF_OBJECT_SHOWPOSTION = 0x40000
EXGF_OBJECT_SHOWRECTBORDER = 0x20000
EXGF_RENDER_CANVAS_ALIAS = 0x40
EXGF_RENDER_METHOD_D2D = 0x100
EXGF_RENDER_METHOD_D2D_GDI_COMPATIBLE = 0x300
EXGF_TEXT_ANTIALIAS = 0x800
EXGF_TEXT_CLEARTYPE = 0x400
EXR_DEFAULT = 0
EXR_LAYOUT = 2
EXR_STRING = 1
GROUPBOX_RADIUS = 1
GROUPBOX_STROKEWIDTH = 2
GROUPBOX_TEXT_OFFSET = 0
GW_CHILD = 5
GW_HWNDNEXT = 2
GW_HWNDPREV = 3
ILVM_SETITEMSIZE = 11001
IMAGE_BITMAP = 0
IMAGE_ICON = 1
LVHT_NOWHERE = 1
LVHT_ONITEM = 14
LVM_CALCITEMSIZE = 5150
LVM_CANCELTHEME = 5151
LVM_DELETEALLCOLUMN = 4900
LVM_DELETEALLITEMS = 4105
LVM_DELETECOLUMN = 4124
LVM_DELETEITEM = 4104
LVM_ENSUREVISIBLE = 4115
LVM_GETCOLUMN = 4121
LVM_GETCOLUMNCOUNT = 4901
LVM_GETCOLUMNTEXT = 4905
LVM_GETCOLUMNWIDTH = 4125
LVM_GETCOUNTPERPAGE = 4136
LVM_GETHOTITEM = 4157
LVM_GETIMAGELIST = 4098
LVM_GETITEM = 4101
LVM_GETITEMCOUNT = 4100
LVM_GETITEMHEIGHT = 4909
LVM_GETITEMRECT = 4110
LVM_GETITEMSTATE = 4140
LVM_GETITEMTEXT = 4141
LVM_GETSELECTEDCOUNT = 4146
LVM_GETSELECTIONMARK = 4162
LVM_GETTOPINDEX = 4135
LVM_HITTEST = 4114
LVM_INSERTCOLUMN = 4123
LVM_INSERTITEM = 4103
LVM_REDRAWITEMS = 4117
LVM_SETCOLUMN = 4122
LVM_SETCOLUMNTEXT = 4904
LVM_SETCOLUMNWIDTH = 4126
LVM_SETIMAGELIST = 4099
LVM_SETITEM = 4102
LVM_SETITEMCOUNT = 4143
LVM_SETITEMHEIGHT = 4908
LVM_SETITEMSTATE = 4139
LVM_SETITEMTEXT = 4142
LVM_SETSELECTIONMARK = 4163
LVM_SORTITEMS = 4144
LVM_UPDATE = 4138
LVN_HOTTRACK = -121
LVN_ITEMCHANGED = -101
MBM_POPUP = 103001
MBN_POPUP = 102401
NM_CALCSIZE = -97
NM_CHAR = -18
NM_CHECK = -89
NM_CLICK = -2
NM_CREATE = -99
NM_CUSTOMDRAW = -12
NM_DBLCLK = -3
NM_DESTROY = -98
NM_EASING = -86
NM_ENABLE = -94
NM_FONTCHANGED = -23
NM_HOVER = -13
NM_INTDLG = -87
NM_KEYDOWN = -15
NM_KILLFOCUS = -8
NM_LDOWN = -20
NM_LEAVE = -91
NM_LUP = -92
NM_MOVE = -96
NM_NCHITTEST = -14
NM_RCLICK = -5
NM_RDBLCLK = -6
NM_RDOWN = -21
NM_RELEASEDCAPTURE = -16
NM_RETURN = -4
NM_SETFOCUS = -7
NM_SHOW = -93
NM_SIZE = -95
NM_TIMER = -90
NM_TOOLTIPSCREATED = -19
NM_TRAYICON = -88
PBF_FILLED = 0
PBF_HOLLOW = 1
PBL_BARCOLOR = 4
PBL_BKCOLOR = 3
PBL_POS = 0
PBL_RADIUS = 2
PBL_RANGE = 1
PBM_GETPOS = 1032
PBM_GETRANGE = 1031
PBM_SETBARCOLOR = 1033
PBM_SETBKCOLOR = 8193
PBM_SETPOS = 1026
PBM_SETRADIUS = 1027
PBM_SETRANGE = 1025
PFA_CENTER = 3
PFA_LEFT = 1
PFA_RIGHT = 2
PFM_ALIGNMENT = 0x00000008
PFM_NUMBERING = 0x00000020
PFM_OFFSET = 0x00000004
PFM_RIGHTINDENT = 0x00000002
PFM_STARTINDENT = 0x00000001
PFN_ARABIC = 2
PFN_BULLET = 1
PFN_LCLETTER = 3
PFN_LCROMAN = 5
PFN_UCLETTER = 4
PFN_UCROMAN = 6
RGN_COMBINE_EXCLUDE = 3
RGN_COMBINE_INTERSECT = 1
RGN_COMBINE_UNION = 0
RGN_COMBINE_XOR = 2
RLVM_CHECK = 99001
RLVM_GETCHECK = 99003
RLVM_GETHITCOL = 99004
RLVM_SETCHECK = 99002
RLVN_CHECK = 97003
RLVN_COLUMNCLICK = 97000
RLVN_DELETE_ITEM = 97004
RLVN_DRAW_TD = 97002
RLVN_DRAW_TR = 97001
SBL_BLOCK_POINT = 3
SBL_BLOCK_SIZE = 4
SBL_MAX = 1
SBL_MIN = 0
SBL_POS = 2
SBM_GETBLOCKRECT = 10010
SBM_PT2VALUE = 10011
SBN_VALUE = 10010
SB_BOTH = 3
SB_CTL = 2
SB_HORZ = 0
SB_VERT = 1
STATE_ALLOWDRAG = 0x40000
STATE_ALLOWFOCUS = 0x100000
STATE_ALLOWMULTIPLE = 0x1000000
STATE_ALLOWSELECT = 0x200000
STATE_ALLOWSIZE = 0x20000
STATE_ANIMATING = 0x4000
STATE_BUSY = 0x800
STATE_CHECKED = 0x10
STATE_DEFAULT = 0x100
STATE_DISABLE = 0x1
STATE_DOWN = 0x8
STATE_FOCUS = 0x4
STATE_HALFSELECT = 0x20
STATE_HIDDEN = 0x8000
STATE_HOVER = 0x80
STATE_HYPERLINK_HOVER = 0x400000
STATE_HYPERLINK_VISITED = 0x800000
STATE_NORMAL = 0
STATE_PASSWORD = 0x2000000
STATE_READONLY = 0x40
STATE_ROLLING = 0x2000
STATE_SELECT = 0x2
STATE_SUBITEM_HIDDEN = 0x400
STATE_SUBITEM_VISIABLE = 0x200
SW_HIDE = 0
SW_SHOW = 5
SW_SHOWMAXIMIZED = 3
SW_SHOWMINIMIZED = 2
SW_SHOWNOACTIVATE = 4
TLVM_GETITEMOBJ = 10021
TLVM_ITEM_CREATE = 10010
TLVM_ITEM_CREATED = 10011
TLVM_ITEM_DESTROY = 10012
TLVM_ITEM_FILL = 10013
TLVM_SETTEMPLATE = 10020
TVGN_CHILD = 4
TVGN_NEXT = 1
TVGN_NEXTVISIBLE = 6
TVGN_PARENT = 3
TVGN_PREVIOUS = 2
TVGN_ROOT = 0
TVHT_NOWHERE = 1
TVHT_ONITEMICON = 2
TVHT_ONITEMINDENT = 8
TVHT_ONITEMLABEL = 4
TVHT_ONITEMSTATEICON = 64
TVI_FIRST = -65535
TVI_LAST = -65534
TVI_ROOT = -65536
TVI_SORT = -65533
TVM_DELETEITEM = 4353
TVM_ENSUREVISIBLE = 4372
TVM_EXPAND = 4354
TVM_GETCOUNT = 4357
TVM_GETIMAGELIST = 4360
TVM_GETINDENT = 4358
TVM_GETITEM = 4364
TVM_GETITEMHEIGHT = 5092
TVM_GETITEMRECT = 4356
TVM_GETITEMTEXTW = 14415
TVM_GETNEXTITEM = 4362
TVM_GETNODEFROMINDEX = 5093
TVM_GETVISIBLECOUNT = 4368
TVM_HITTEST = 4369
TVM_INSERTITEM = 4352
TVM_SELECTITEM = 4363
TVM_SETIMAGELIST = 4361
TVM_SETINDENT = 4359
TVM_SETITEM = 4365
TVM_SETITEMHEIGHT = 5091
TVM_SETITEMTEXTW = 14414
TVM_UPDATE = 4499
TVN_DELETEITEM = 391
TVN_DRAWITEM = 3099
TVN_ITEMEXPANDED = 394
TVN_ITEMEXPANDING = 395
UNIT_PERCENT = 1
UNIT_PIXEL = 0
WM_EX_DROP = -9
WM_EX_EASING = -8
WM_EX_EXITPOPUP = -7
WM_EX_INITPOPUP = -6
WM_EX_JS_DISPATCH = -2
WM_EX_LCLICK = -3
WM_EX_MCLICK = -5
WM_EX_PAINTING = -10
WM_EX_PROPS = -11
WM_EX_RCLICK = -4
WM_EX_XML_PROPDISPATCH = -1


class EX_PAINTSTRUCT(ctypes.Structure):
    """
    绘制信息结构
    """
    _fields_ = [
        ('hcanvas', c_int),
        ('htheme', c_void_p),
        ('style', c_int),
        ('styleex', c_int),
        ('text_formet', c_int),
        ('state', c_int),
        ('owner_data', c_void_p),
        ('width', c_uint),
        ('height', c_uint),
        ('rcpaint_left', c_int),
        ('rcpaint_top', c_int),
        ('rcpaint_right', c_int),
        ('rcpaint_bottom', c_int),
        ('rctext_left', c_int),
        ('rctext_top', c_int),
        ('rctext_right', c_int),
        ('rctext_bottom', c_int),
        ('reserved', c_void_p)]


class EX_CUSTOMDRAW(ctypes.Structure):
    """
    自定义绘制信息结构
    """
    _fields_ = [
        ('hcanvas', c_int),
        ('htheme', c_void_p),
        ('state', c_int),
        ('style', c_int),
        ('rcpaint_left', c_int),
        ('rcpaint_top', c_int),
        ('rcpaint_right', c_int),
        ('rcpaint_bottom', c_int),
        ('item', c_int),
        ('itemparam', c_size_t)
    ]


class EX_NMHDR(ctypes.Structure):
    """
    接收WM_NOTIFY通知信息结构
    """
    _fields_ = [
        ('hObjFrom', c_int),
        ('idFrom', c_int),
        ('nCode', c_int),
        ('wParam', c_size_t),
        ('lParam', c_longlong)
    ]


class EX_EASINGINFO(ctypes.Structure):
    """
    缓动信息结构
    """
    _pack_ = 1
    _fields_ = [
        ('pEasing', c_void_p),
        ('nProgress', c_double),
        ('nCurrent', c_double),
        ('pEasingContext', c_void_p),
        ('nTimesSurplus', c_int),
        ('p1', c_longlong),
        ('p2', c_longlong),
        ('p3', c_longlong),
        ('p4', c_longlong)
    ]


class EX_REPORTLIST_COLUMNINFO(ctypes.Structure):
    """
    报表列信息结构
    """
    _fields_ = [
        ('wzText', c_wchar_p),
        ('nWidth', c_int),
        ('dwStyle', c_int),
        ('dwTextFormat', c_int),
        ('crText', c_int),
        ('nInsertIndex', c_int)
    ]


class EX_REPORTLIST_ITEMINFO(ctypes.Structure):
    """
    报表项目信息结构
    """
    _fields_ = [
        ('iRow', c_int),
        ('iCol', c_int),
        ('dwStyle', c_int),
        ('wzText', c_wchar_p),
        ('nImageIndex', c_int),
        ('lParam', c_longlong),
        ('dwState', c_int)
    ]


class EX_REPORTLIST_ROWINFO(ctypes.Structure):
    """
    报表行信息结构
    """
    _fields_ = [
        ('nInsertIndex', c_int),
        ('dwStyle', c_int),
        ('lParam', c_longlong),
        ('nImageIndex', c_int)
    ]


class EX_TREEVIEW_NODEITEM(ctypes.Structure):
    """
    树形框节点信息结构
    """
    _fields_ = [
        ('nID', c_int),
        ('lpTitle', c_wchar_p),
        ('lParam', c_longlong),
        ('nImageIndex', c_int),
        ('nImageIndexExpand', c_int),
        ('fExpand', c_bool),
        ('dwStyle', c_int),
        ('nDepth', c_int),
        ('pParent', c_void_p),
        ('pPrev', c_void_p),
        ('pNext', c_void_p),
        ('pChildFirst', c_void_p),
        ('nCountChild', c_int)
    ]


class EX_TREEVIEW_INSERTINFO(ctypes.Structure):
    """
    树形框插入项目信息结构
    """
    _fields_ = [
        ('itemParent', c_void_p),
        ('itemInsertAfter', c_void_p),
        ('nID', c_int),
        ('tzText', c_wchar_p),
        ('lParam', c_longlong),
        ('nImageIndex', c_int),
        ('nImageIndexExpand', c_int),
        ('fExpand', c_bool),
        ('dwStyle', c_int),
        ('fUpdateLater', c_bool)
    ]


class EX_BACKGROUNDIMAGEINFO(ctypes.Structure):
    """
    背景信息结构
    """
    _fields_ = [
        ('dwFlags', c_int),
        ('hImage', c_int),
        ('x', c_int),
        ('y', c_int),
        ('dwRepeat', c_int),
        ('lpGrid', c_void_p),
        ('lpDelay', c_void_p),
        ('curFrame', c_int),
        ('maxFrame', c_int),
        ('dwAlpha', c_int)
    ]


class EX_CLASSINFO(ctypes.Structure):
    """
    组件类信息结构
    """
    _fields_ = [
        ('dwFlags', c_int),
        ('dwStyle', c_int),
        ('dwStyleEx', c_int),
        ('dwTextFormat', c_int),
        ('cbObjExtra', c_int),
        ('hCursor', c_size_t),
        ('pfnClsProc', c_void_p),
        ('atomName', c_int)
    ]


class EX_OBJ_PROPS(ctypes.Structure):
    """
    扩展控件属性信息结构
    """
    _fields_ = [
        ('cr_bkg_normal', c_int),
        ('cr_bkg_hover', c_int),
        ('cr_bkg_downorcheck', c_int),
        ('cr_bkg_begin', c_int),
        ('cr_bkg_end', c_int),
        ('cr_border_normal', c_int),
        ('cr_border_hover', c_int),
        ('cr_border_downorcheck', c_int),
        ('cr_border_begin', c_int),
        ('cr_border_ebd', c_int),
        ('cr_icon_normal', c_int),
        ('cr_icon_hover', c_int),
        ('cr_icon_downorfocus', c_int),
        ('radius', c_int),
        ('strokeWidth', c_int),
        ('icon_position', c_int)
    ]


class EX_ICONLISTVIEW_ITEMINFO(ctypes.Structure):
    """
    图标列表框插入信息结构
    """
    _fields_ = [
        ('nIndex', c_int),
        ('nImageIndex', c_int),
        ('pwzText', c_wchar_p)
    ]


class EX_IMAGEINFO(ctypes.Structure):
    """
    图像属性信息
    """
    _fields_ = [
        ('imgNormal', c_int),
        ('imgHover', c_int),
        ('imgDownOrChecked', c_int)
    ]


class EX_DROPINFO(ctypes.Structure):
    """
    拖曳信息结构
    """
    _fields_ = [
        ('pDataObject', c_void_p),
        ('grfKeyState', c_int),
        ('x', c_int),
        ('y', c_int)
    ]


class EX_CHARRANGE(ctypes.Structure):
    """
    富文本框EM_EXSETSEL消息lParam参数结构
    """
    _fields_ = [
        ('cpMin', c_int),
        ('cpMax', c_int)
    ]


class EX_TEXTRANGE(ctypes.Structure):
    """
    富文本框EM_GETTEXTRANGE,EM_FINDTEXT消息接收lParam参数
    """
    _fields_ = [
        ('chrg', EX_CHARRANGE),
        ('pwzText', c_wchar_p)
    ]


class NMHDR(ctypes.Structure):
    """
    NMHDR
    """
    _fields_ = [
        ('hwndFrom', c_size_t),
        ('idFrom', c_longlong),
        ('code', c_int),
    ]


class EX_SELCHANGE(ctypes.Structure):
    """
    富文本框EN_SELCHANGE消息lParam参数结构
    """
    _fields_ = [
        ('nmhdr', NMHDR),
        ('chrg', EX_CHARRANGE),
        ('seltyp', c_short)
    ]


class EX_ENLINK(ctypes.Structure):
    """
    富文本框EN_LINK消息lParam参数结构
    """
    _pack_ = 1
    _fields_ = [
        ('nmhdr', NMHDR),
        ('msg', c_int),
        ('wParam', c_size_t),
        ('lParam', c_longlong),
        ('chrg', EX_CHARRANGE)
    ]


class EX_SETTEXTEX(ctypes.Structure):
    """
    富文本框替换文本信息结构
    """
    _fields_ = [
        ('flags', c_int),
        ('codePage', c_int)
    ]


def lobyte(w):
    return w & 0xff


def hibyte(w):
    return (w >> 8) & 0xff


def loword(l):
    return l & 0xffff


def hiword(l):
    return (l >> 16) & 0xffff


class ExDUIR(object):
    def __init__(self, flag: int = EXGF_RENDER_METHOD_D2D):
        with open(EXT_PATH, 'rb') as f:
            ext = f.read()
        libexdui.Ex_Init(0, flag, 0, 0, ext, len(ext), 0, 0)

    def brushCreate(self, argb: int) -> int:
        libexdui._brush_create.restype = c_void_p
        return libexdui._brush_create(argb)

    def brushCreateFromCanvas(self, hCanvas: int) -> int:
        libexdui._brush_createfromcanvas.restype = c_void_p
        return libexdui._brush_createfromcanvas(hCanvas)

    def brushCreateFromCanvas2(self, hCanvas: int, alpha: int) -> int:
        libexdui._brush_createfromcanvas2.restype = c_void_p
        return libexdui._brush_createfromcanvas2(hCanvas, alpha)

    def brushCreateFromImg(self, hImg: int) -> int:
        libexdui._brush_createfromimg.restype = c_void_p
        return libexdui._brush_createfromimg(hImg)

    def brushCreateLinear(self, xStart: float, yStart: float, xEnd: float, yEnd: float, crBegin: int, crEnd: int) -> int:
        libexdui._brush_createlinear.argtypes = (
            c_float, c_float, c_float, c_float, c_int, c_int)
        libexdui._brush_createlinear.restype = c_void_p
        return libexdui._brush_createlinear(xStart, yStart, xEnd, yEnd, crBegin, crEnd)

    def brushCreateLinearEx(self, xStart: float, yStart: float, xEnd: float, yEnd: float, arrStopPts: int, cStopPts: int) -> int:
        libexdui._brush_createlinear_ex.argtypes = (
            c_float, c_float, c_float, c_float, c_void_p, c_int)
        libexdui._brush_createlinear_ex.restype = c_void_p
        return libexdui._brush_createlinear_ex(xStart, yStart, xEnd, yEnd, arrStopPts, cStopPts)

    def brushDestroy(self, hBrush: int) -> bool:
        libexdui._brush_destroy.argtypes = (c_void_p,)
        return libexdui._brush_destroy(hBrush)

    def brushSetColor(self, hBrush: int, argb: int) -> int:
        libexdui._brush_setcolor.argtypes = (c_void_p, c_int)
        return libexdui._brush_setcolor(hBrush, argb)

    def brushSetTransform(self, hBrush: int, matrix: int) -> None:
        libexdui._brush_settransform.argtypes = (c_void_p, c_void_p)
        libexdui._brush_settransform(hBrush, matrix)

    def canvasBeginDraw(self, hCanvas: int) -> bool:
        return libexdui._canvas_begindraw(hCanvas)

    def canvasBlur(self, hCanvas: int, fDeviation: float, lprc: int) -> bool:
        libexdui._canvas_blur.argtypes = (c_int, c_float, c_void_p)
        return libexdui._canvas_blur(hCanvas, fDeviation, lprc)

    def canvasCalcTextSize(self, hCanvas: int, hFont: int, lpwzText: str, dwLen: int, dwDTFormat: int, lParam: int, layoutWidth: float, layoutHeight: float, lpWidth: int, lpHeight: int) -> bool:
        libexdui._canvas_calctextsize.argtypes = (
            c_int, c_int, c_wchar_p, c_longlong, c_int, c_longlong, c_float, c_float, c_void_p, c_void_p)
        return libexdui._canvas_calctextsize(hCanvas, hFont, lpwzText, dwLen, dwDTFormat, lParam, layoutWidth, layoutHeight, lpWidth, lpHeight)

    def canvasClear(self, hCanvas: int, Color: int) -> bool:
        return libexdui._canvas_clear(hCanvas, Color)

    def canvasClipRect(self, hCanvas: int, left: int, top: int, right: int, bottom: int) -> bool:
        return libexdui._canvas_cliprect(hCanvas, left, top, right, bottom)

    def canvasCreateFromExdui(self, hExDui: int, width: int, height: int, dwFlags: int) -> int:
        return libexdui._canvas_createfromexdui(hExDui, width, height, dwFlags)

    def canvasCreateFromObj(self, hObj: int, uWidth: int, uHeight: int, dwFlags: int) -> int:
        libexdui._canvas_createfromobj.argtypes = (c_int, c_int, c_int, c_int)
        return libexdui._canvas_createfromobj(hObj, uWidth, uHeight, dwFlags)

    def canvasDestroy(self, hCanvas: int) -> bool:
        return libexdui._canvas_destroy(hCanvas)

    def canvasDrawCanvas(self, hCanvas: int, sCanvas: int, dstLeft: int, dstTop: int, dstRight: int, dstBottom: int, srcLeft: int, srcTop: int, dwAlpha: int, dwCompositeMode: int) -> bool:
        return libexdui._canvas_drawcanvas(hCanvas, sCanvas, dstLeft, dstTop, dstRight, dstBottom, srcLeft, srcTop, dwAlpha, dwCompositeMode)

    def canvasDrawEllipse(self, hCanvas: int, hBrush: int, x: float, y: float, radiusX: float, radiusY: float, strokeWidth: float, strokeStyle: int) -> bool:
        libexdui._canvas_drawellipse.argtypes = (
            c_int, c_void_p, c_float, float, float, float, c_float, c_int)
        return libexdui._canvas_drawellipse(hCanvas, hBrush, x, y, radiusX, radiusY, strokeWidth, strokeStyle)

    def canvasDrawImage(self, hCanvas: int, hImage: int, Left: float, Top: float, alpha: int) -> bool:
        libexdui._canvas_drawimage.argtypes = (
            c_int, c_int, c_float, c_float, c_int)
        return libexdui._canvas_drawimage(hCanvas, hImage, Left, Top, alpha)

    def canvasDrawImageFromGrid(self, hCanvas: int, hImage: int, dstLeft: float, dstTop: float, dstRight: float, dstBottom: float,
                                srcLeft: float, srcTop: float, srcRight: float, srcBottom: float,
                                gridPaddingLeft: float, gridPaddingTop: float, gridPaddingRight: float, gridPaddingBottom: float, flags: int, alpha: int) -> bool:
        libexdui._canvas_drawimagefromgrid.argtypes = (
            c_int, c_int, c_float, c_float, c_float, c_float, c_float, c_float, c_float, c_float, c_float, c_float, c_float, c_float, c_int, c_int)
        return libexdui._canvas_drawimagefromgrid(hCanvas, hImage, dstLeft, dstTop, dstRight, dstBottom, srcLeft, srcTop, srcRight, srcBottom, gridPaddingLeft, gridPaddingTop, gridPaddingRight, gridPaddingBottom, flags, alpha)

    def canvasDrawImageRect(self, hCanvas: int, hImage: int, Left: float, Top: float, Right: float, Bottom: float, alpha: int) -> bool:
        libexdui._canvas_drawimagerect.argtypes = (
            c_int, c_int, c_float, c_float, c_float, c_float, c_int)
        return libexdui._canvas_drawimagerect(hCanvas, hImage, Left, Top, Right, Bottom, alpha)

    def canvasDrawImageRectRect(self, hCanvas: int, hImage: int, dstLeft: float, dstTop: float, dstRight: float, dstBottom: float, srcLeft: float, srcTop: float, srcRight: float, srcBottom: float, alpha: int) -> bool:
        libexdui._canvas_drawimagerectrect.argtypes = (
            c_int, c_int, c_float, c_float, c_float, c_float, c_float, c_float, c_float, c_float, c_int)
        return libexdui._canvas_drawimagerectrect(hCanvas, hImage, dstLeft, dstTop, dstRight, dstBottom, srcLeft, srcTop, srcRight, srcBottom, alpha)

    def canvasDrawLine(self, hCanvas: int, hBrush: int, X1: float, Y1: float, X2: float, Y2: float, strokeWidth: float, strokeStyle: int) -> bool:
        libexdui._canvas_drawline.argtypes = (
            c_int, c_void_p, c_float, c_float, c_float, c_float, c_float, c_int)
        return libexdui._canvas_drawline(hCanvas, hBrush, X1, Y1, X2, Y2, strokeWidth, strokeStyle)

    def canvasDrawPath(self, hCanvas: int, hPath: int, hBrush: int, strokeWidth: float, strokeStyle: int) -> bool:
        libexdui._canvas_drawpath.argtypes = (
            c_int, c_int, c_void_p, c_float, c_int)
        return libexdui._canvas_drawpath(hCanvas, hPath, hBrush, strokeWidth, strokeStyle)

    def canvasDrawRect(self, hCanvas: int, hBrush: int, left: float, top: float, right: float, bottom: float, strokeWidth: float, strokeStyle: int) -> bool:
        libexdui._canvas_drawrect.argtypes = (
            c_int, c_void_p, c_float, c_float, c_float, c_float, c_float, c_int)
        return libexdui._canvas_drawrect(hCanvas, hBrush, left, top, right, bottom, strokeWidth, strokeStyle)

    def canvasDrawRoundedRect(self, hCanvas: int, hBrush: int, left: float, top: float, right: float, bottom: float, radiusX: float, radiusY: float, strokeWidth: float, strokeStyle: int) -> bool:
        libexdui._canvas_drawroundedrect.argtypes = (
            c_int, c_void_p, c_float, c_float, c_float, c_float, c_float, c_float, c_float, c_int)
        return libexdui._canvas_drawroundedrect(hCanvas, hBrush, left, top, right, bottom, radiusX, radiusY, strokeWidth, strokeStyle)

    def canvasDrawText(self, hCanvas: int, hFont: int, crText: int, lpwzText: str, dwLen: int, dwDTFormat: int, left: float, top: float, right: float, bottom: float) -> bool:
        libexdui._canvas_drawtext.argtypes = (
            c_int, c_int, c_int, c_wchar_p, c_int, c_int, c_float, c_float, c_float, c_float)
        return libexdui._canvas_drawtext(hCanvas, hFont, crText, lpwzText, dwLen, dwDTFormat, left, top, right, bottom)

    def canvasDrawText2(self, hCanvas: int, hFont: int, hBrush: int, lpwzText: str, dwLen: int, dwDTFormat: int, left: float, top: float, right: float, bottom: float) -> bool:
        libexdui._canvas_drawtext2.argtypes = (
            c_int, c_int, c_void_p, c_wchar_p, c_int, c_int, c_float, c_float, c_float, c_float)
        return libexdui._canvas_drawtext2(hCanvas, hFont, hBrush, lpwzText, dwLen, dwDTFormat, left, top, right, bottom)

    def canvasDrawTextEx(self, hCanvas: int, hFont: int, crText: int, lpwzText: str, dwLen: int, dwDTFormat: int, left: float, top: float, right: float, bottom: float, iGlowsize: int, crShadom: int, lParam: int, prclayout: int) -> bool:
        libexdui._canvas_drawtextex.argtypes = (
            c_int, c_int, c_int, c_wchar_p, c_int, c_int, c_float, c_float, c_float, c_float, c_int, c_int, c_longlong, c_void_p)
        return libexdui._canvas_drawtextex(hCanvas, hFont, crText, lpwzText, dwLen, dwDTFormat, left, top, right, bottom, iGlowsize, crShadom, lParam, prclayout)

    def canvasEndDaw(self, hCanvas: int) -> bool:
        libexdui._canvas_enddraw.argtypes = (c_int,)
        return libexdui._canvas_enddraw(hCanvas)

    def canvasFillEllipse(self, hCanvas: int, hBrush: int, x: float, y: float, radiusX: float, radiusY: float) -> bool:
        libexdui._canvas_fillellipse.argtypes = (
            c_int, c_void_p, c_float, c_float, c_float, c_float)
        return libexdui._canvas_fillellipse(hCanvas, hBrush, x, y, radiusX, radiusY)

    def canvasFillPath(self, hCanvas: int, hPath: int, hBrush: int) -> bool:
        libexdui._canvas_fillpath.argtypes = (c_int, c_int, c_void_p)
        return libexdui._canvas_fillpath(hCanvas, hPath, hBrush)

    def canvasFillRect(self, hCanvas: int, hBrush: int, left: float, top: float, right: float, bottom: float) -> bool:
        libexdui._canvas_fillrect.argtypes = (
            c_int, c_void_p, c_float, c_float, c_float, c_float)
        return libexdui._canvas_fillrect(hCanvas, hBrush, left, top, right, bottom)

    def canvasFillRegion(self, hCanvas: int, hRgn: int, hBrush: int) -> bool:
        libexdui._canvas_fillregion.argtypes = (c_int, c_void_p, c_void_p)
        return libexdui._canvas_fillregion(hCanvas, hRgn, hBrush)

    def canvasFillRoundedImage(self, hCanvas: int, hImg: int, left: float, top: float, Width: float, Height: float, RadiuX: float, RadiuY: float, shadowNum: int, number: int, crShadow: int) -> bool:
        libexdui._canvas_fillroundedimage.argtypes = (
            c_int, c_int, c_float, c_float, c_float, c_float, c_float, c_float, c_void_p, c_int, c_int)
        return libexdui._canvas_fillroundedimage(hCanvas, hImg, left, top, Width, Height, RadiuX, RadiuY, shadowNum, number, crShadow)

    def canvasFillRoundedRect(self, hCanvas: int, hBrush: int, left: float, top: float, right: float, bottom: float, radiusX: float, radiusY: float) -> bool:
        libexdui._canvas_fillroundedrect.argtypes = (
            c_int, c_void_p, c_float, c_float, c_float, c_float, c_float, c_float)
        return libexdui._canvas_fillroundedrect(hCanvas, hBrush, left, top, right, bottom, radiusX, radiusY)

    def canvasFlush(self, hCanvas: int) -> bool:
        libexdui._canvas_flush.argtypes = (c_int,)
        return libexdui._canvas_flush(hCanvas)

    def canvasGetContext(self, hCanvas: int, nType: int) -> int:
        libexdui._canvas_getcontext.argtypes = (c_int, c_int)
        libexdui._canvas_getcontext.restype = c_void_p
        return libexdui._canvas_getcontext(hCanvas, nType)

    def canvasGetDC(self, hCanvas: int) -> int:
        libexdui._canvas_getdc.argtypes = (c_int, c_int)
        libexdui._canvas_getdc.restype = c_void_p
        return libexdui._canvas_getdc(hCanvas)

    def canvasGetSize(self, hCanvas: int, width: int, height: int) -> bool:
        libexdui._canvas_getsize.argtypes = (c_int, c_void_p, c_void_p)
        return libexdui._canvas_getsize(hCanvas, width, height)

    def canvasReleaseDc(self, hCanvas: int) -> bool:
        libexdui._canvas_releasedc.argtypes = (c_int,)
        return libexdui._canvas_releasedc(hCanvas)

    def canvasResetClip(self, hCanvas: int) -> bool:
        libexdui._canvas_resetclip.argtypes = (c_int,)
        return libexdui._canvas_resetclip(hCanvas)

    def canvasResize(self, hCanvas: int, width: int, height: int) -> bool:
        libexdui._canvas_resize.argtypes = (c_int, c_int, c_int)
        return libexdui._canvas_resize(hCanvas, width, height)

    def canvasRotateHue(self, hCanvas: int, fAngle: float) -> bool:
        libexdui._canvas_rotate_hue.argtypes = (c_int, c_float)
        return libexdui._canvas_rotate_hue(hCanvas, fAngle)

    def canvasSetantialias(self, hCanvas: int, antialias: bool) -> bool:
        return libexdui._canvas_setantialias(hCanvas, antialias)

    def canvasSetImageAntialias(self, hCanvas: int, antialias: bool) -> bool:
        return libexdui._canvas_setimageantialias(hCanvas, antialias)

    def canvasSetTextAntialiasMode(self, hCanvas: int, antialias: bool) -> bool:
        return libexdui._canvas_settextantialiasmode(hCanvas, antialias)

    def canvasSetTransform(self, hCanvas: int, pMatrix: int) -> bool:
        libexdui._canvas_settransform.argtypes = (c_int, c_void_p)
        return libexdui._canvas_settransform(hCanvas, pMatrix)

    def easingCreate(self, dwType: int, pEasingContext: int, dwMode: int, pContext: int, nTotalTime: int, nInterval: int, nState: int, nStart: int, nStop: int, param1: int, param2: int, param3: int, param4: int) -> int:
        libexdui._easing_create.argtypes = (c_int, c_void_p, c_int, c_longlong, c_int,
                                            c_int, c_int, c_int, c_int, c_longlong, c_longlong, c_longlong, c_longlong)
        libexdui._easing_create.restype = c_void_p
        return libexdui._easing_create(dwType, pEasingContext, dwMode, pContext, nTotalTime, nInterval, nState, nStart, nStop, param1, param2, param3, param4)

    def easingGetState(self, pEasing: int) -> int:
        libexdui._easing_getstate.argtypes = (c_void_p,)
        return libexdui._easing_getstate(pEasing)

    def easingSetState(self, pEasing: int, nState: int) -> bool:
        libexdui._easing_setstate.argtypes = (c_void_p, c_int)
        return libexdui._easing_setstate(pEasing, nState)

    def fontCreate(self) -> int:
        return libexdui._font_create()

    def fontCreateFromFamily(self, lpwzFontFace: str, dwFontSize: int, dwFontStyle: int) -> int:
        libexdui._font_createfromfamily.argtypes = (c_wchar_p, c_int, c_int)
        return libexdui._font_createfromfamily(lpwzFontFace, dwFontSize, dwFontStyle)

    def fontCreateFromLogFont(self, lpLogfont: int) -> int:
        libexdui._font_createfromlogfont.argtypes = (c_void_p,)
        return libexdui._font_createfromlogfont(lpLogfont)

    def fontDestroy(self, hFont: int) -> bool:
        return libexdui._font_destroy(hFont)

    def fontGetContext(self, hFont: int) -> int:
        libexdui._font_getcontext.restype = c_void_p
        return libexdui._font_getcontext(hFont)

    def fontGetLogFont(self, hFont: int, lpLogFont: int) -> bool:
        libexdui._font_getlogfont.argtypes = (c_int, c_void_p)
        return libexdui._font_getlogfont(hFont, lpLogFont)

    def imgChangeColor(self, hImg: int, argb: int) -> bool:
        libexdui._img_changecolor.argtypes = (c_int, c_int)
        return libexdui._img_changecolor(hImg, argb)

    def imgClip(self, hImg: int, left: int, top: int, width: int, height: int, phImg: int) -> bool:
        libexdui._img_clip.argtypes = (c_int, c_int, c_int, c_int, c_void_p)
        return libexdui._img_clip(hImg, left, top, width, height, phImg)

    def imgCopy(self, hImg: int, phImg: int) -> bool:
        libexdui._img_copy.argtypes = (c_int, c_void_p)
        return libexdui._img_copy(hImg, phImg)

    def imgCopyRect(self, hImg: int, x: int, y: int, width: int, height: int, phImg: int) -> bool:
        libexdui._img_copyrect.argtypes = (
            c_int, c_int, c_int, c_int, c_int, c_void_p)
        return libexdui._img_copyrect(hImg, x, y, width, height, phImg)

    def imgCreate(self, width: int, height: int, phImg: int) -> bool:
        libexdui._img_create.argtypes = (c_int, c_int, c_void_p)
        return libexdui._img_create(width, height, phImg)

    def imgCreateFromFile(self, lpwzFilename: str, phImg: int) -> bool:
        libexdui._img_createfromfile.argtypes = (c_wchar_p, c_void_p)
        return libexdui._img_createfromfile(lpwzFilename, phImg)

    def imgCreateFromhBitmap(self, hBitmap: int, hPalette: int, fPreAlpha: bool, phImg: int) -> bool:
        libexdui._img_createfromhbitmap.argtypes = (
            c_void_p, c_void_p, bool, c_void_p)
        return libexdui._img_createfromhbitmap(hBitmap, hPalette, fPreAlpha, phImg)

    def imgCreateFromhIcon(self, hIcon: int, phImg: int) -> bool:
        libexdui._img_createfromhicon.argtypes = (c_size_t, c_void_p)
        return libexdui._img_createfromhicon(hIcon, phImg)

    def imgCreateFromMemory(self, lpData: int, dwLen: int, phImg: int) -> bool:
        libexdui._img_createfrommemory.argtypes = (c_void_p, c_int, c_void_p)
        return libexdui._img_createfrommemory(lpData, dwLen, phImg)

    def imgCreateFromRes(self, hRes: int, atomPath: int, phImg: int) -> bool:
        libexdui._img_createfromres.argtypes = (c_void_p, c_int, c_void_p)
        return libexdui._img_createfromres(hRes, atomPath, phImg)

    def imgCreateFromStream(self, lpStream: int, phImg: int) -> bool:
        libexdui._img_createfromstream.argtypes = (c_void_p, c_void_p)
        return libexdui._img_createfromstream(lpStream, phImg)

    def imgDestroy(self, hImg: int) -> bool:
        return libexdui._img_destroy(hImg)

    def imgGetframeCount(self, hImage: int, nFrameCount: int) -> int:
        return libexdui._img_getframecount(hImage, nFrameCount)

    def imgGetFrameDelay(self, hImg: int, lpDelayAry: int, nFrames: int) -> bool:
        libexdui._img_getframedelay.argtypes = (c_int, c_void_p, c_int)
        return libexdui._img_getframedelay(hImg, lpDelayAry, nFrames)

    def imgGetPixel(self, hImg: int, x: int, y: int, retPixel: int) -> bool:
        libexdui._img_getpixel.argtypes = (c_int, c_int, c_int, c_void_p)
        return libexdui._img_getpixel(hImg, x, y, retPixel)

    def imgGetSize(self, hImg: int, lpWidth: int, lpHeight: int) -> bool:
        libexdui._img_getsize.argtypes = (c_int, c_void_p, c_void_p)
        return libexdui._img_getsize(hImg, lpWidth, lpHeight)

    def imgHeight(self, hImg: int) -> int:
        return libexdui._img_height(hImg)

    def imgLock(self, hImg: int, lpRectL: int, flags: int, PixelFormat: int, lpLockedBitmapData: int) -> bool:
        libexdui._img_lock.argtypes = (c_int, c_void_p, c_int, c_int, c_void_p)
        return libexdui._img_lock(hImg, lpRectL, flags, PixelFormat, lpLockedBitmapData)

    def imgRotateFlip(self, hImg: int, rfType: int, phImg: int) -> bool:
        libexdui._img_rotateflip.argtypes = (c_int, c_int, c_void_p)
        return libexdui._img_rotateflip(hImg, rfType, phImg)

    def imgSaveToFile(self, hImg: int, wzFileName: str) -> bool:
        libexdui._img_savetofile.argtypes = (c_int, c_wchar_p)
        return libexdui._img_savetofile(hImg, wzFileName)

    def imgSaveToMemory(self, hImg: int, lpBuffer: int) -> bool:
        libexdui._img_savetomemory.argtypes = (c_int, c_void_p)
        return libexdui._img_savetomemory(hImg, lpBuffer)

    def imgScale(self, hImg: int, width: int, height: int, phImg: int) -> bool:
        libexdui._img_scale.argtypes = (c_int, c_int, c_int, c_void_p)
        return libexdui._img_scale(hImg, width, height, phImg)

    def imgSelectActiveFrame(self, hImg: int, nIndex: int) -> bool:
        return libexdui._img_selectactiveframe(hImg, nIndex)

    def imgSetPixel(self, hImg: int, x: int, y: int, color: int) -> bool:
        return libexdui._img_setpixel(hImg, x, y, color)

    def imgUnlock(self, hImg: int, lpLockedBitmapData: int) -> bool:
        libexdui._img_unlock.argtypes = (c_int, c_void_p)
        return libexdui._img_unlock(hImg, lpLockedBitmapData)

    def imgWidth(self, hImg: int) -> int:
        return libexdui._img_width(hImg)

    def imgListAdd(self, hImageList: int, pImg: int, dwBytes: int, nIndex: int) -> int:
        libexdui._imglist_add.argtypes = (c_void_p, c_void_p, c_int, c_size_t)
        libexdui._imglist_add.restype = c_size_t
        return libexdui._imglist_add(hImageList, pImg, dwBytes, nIndex)

    def imgListAddimage(self, hImageList: int, hImg: int, nIndex: int) -> int:
        libexdui._imglist_addimage.argtypes = (c_void_p, c_int, c_size_t)
        return libexdui._imglist_addimage(hImageList, hImg, nIndex)

    def imgListCount(self, hImageList: int) -> int:
        libexdui._imglist_count.argtypes = (c_void_p,)
        return libexdui._imglist_count(hImageList)

    def imgListCreate(self, width: int, height: int) -> int:
        libexdui._imglist_create.restype = c_void_p
        return libexdui._imglist_create(width, height)

    def imgListDel(self, hImageList: int, nIndex: int) -> bool:
        libexdui._imglist_del.argtypes = (c_void_p, c_size_t)
        return libexdui._imglist_del(hImageList, nIndex)

    def imgListDestroy(self, hImageList: int) -> bool:
        libexdui._imglist_destroy.argtypes = (c_void_p,)
        return libexdui._imglist_destroy(hImageList)

    def imgListDraw(self, hImageList: int, nIndex: int, hCanvas: int, nLeft: int, nTop: int, nRight: int, nBottom: int, nAlpha: int) -> bool:
        libexdui._imglist_draw.argtypes = (
            c_void_p, c_size_t, c_int, c_int, c_int, c_int, c_int, c_int)
        return libexdui._imglist_draw(hImageList, nIndex, hCanvas, nLeft, nTop, nRight, nBottom, nAlpha)

    def imgListGet(self, hImageList: int, nIndex: int) -> int:
        libexdui._imglist_get.argtypes = (c_void_p, c_size_t)
        return libexdui._imglist_get(hImageList, nIndex)

    def imgListSet(self, hImageList: int, nIndex: int, pImg: int, dwBytes: int) -> bool:
        libexdui._imglist_set.argtypes = (
            c_void_p, c_size_t, c_void_p, c_size_t)
        return libexdui._imglist_set(hImageList, nIndex, pImg, dwBytes)

    def imgListSetImage(self, hImageList: int, nIndex: int, hImg: int) -> bool:
        libexdui._imglist_setimage.argtypes = (c_void_p, c_size_t, c_int)
        return libexdui._imglist_setimage(hImageList, nIndex, hImg)

    def imgListSize(self, hImageList: int, pWidth: int, pHeight: int) -> bool:
        libexdui._imglist_size.argtypes = (c_void_p, c_void_p, c_void_p)
        return libexdui._imglist_size(hImageList, pWidth, pHeight)

    def layoutAbsoluteLock(self, hLayout: int, hObjChild: int, tLeft: int, tTop: int, tRight: int, tBottom: int, tWidth: int, tHeight: int) -> bool:
        return libexdui._layout_absolute_lock(hLayout, hObjChild, tLeft, tTop, tRight, tBottom, tWidth, tHeight)

    def layoutAbsoluteSetedge(self, hLayout: int, hObjChild: int, dwEdge: int, dwType: int, nValue: int) -> bool:
        return libexdui._layout_absolute_setedge(hLayout, hObjChild, dwEdge, dwType, nValue)

    def layoutAddChild(self, hLayout: int, hObj: int) -> bool:
        return libexdui._layout_addchild(hLayout, hObj)

    def layoutAddChildren(self, hLayout: int, fDesc: bool, dwObjClassATOM: int, nCount: int) -> bool:
        return libexdui._layout_addchildren(hLayout, fDesc, dwObjClassATOM, nCount)

    def layoutCreate(self, nType: int, hObjBind: int) -> int:
        return libexdui._layout_create(nType, hObjBind)

    def layoutDeleteChild(self, hLayout: int, hObj: int) -> bool:
        return libexdui._layout_deletechild(hLayout, hObj)

    def layoutDeleteChildren(self, hLayout: int, dwObjClassATOM: int) -> bool:
        return libexdui._layout_deletechildren(hLayout, dwObjClassATOM)

    def layoutDestroy(self, hLayout: int) -> bool:
        return libexdui._layout_destroy(hLayout)

    def layoutEnableUpdate(self, hLayout: int, fUpdateable: bool) -> bool:
        return libexdui._layout_enableupdate(hLayout, fUpdateable)

    def layoutGetChildProp(self, hLayout: int, hObj: int, dwPropID: int, pvValue: int) -> bool:
        libexdui._layout_getchildprop.argtypes = (
            c_int, c_int, c_int, c_void_p)
        return libexdui._layout_getchildprop(hLayout, hObj, dwPropID, pvValue)

    def layoutGetChildPropList(self, hLayout: int, hObj: int, lpProps: int) -> bool:
        libexdui._layout_getchildprop.argtypes = (c_int, c_int, c_void_p)
        return libexdui._layout_getchildproplist(hLayout, hObj, lpProps)

    def layoutGetProp(self, hLayout: int, dwPropID: int) -> int:
        return libexdui._layout_getprop(hLayout, dwPropID)

    def layoutGetPropList(self, hLayout: int) -> int:
        libexdui._layout_getproplist.restype = c_void_p
        return libexdui._layout_getproplist(hLayout)

    def layoutGetType(self, hLayout: int) -> int:
        return libexdui._layout_gettype(hLayout)

    def layoutNotify(self, hLayout: int, nEvent: int, wParam, lParam) -> int:
        return libexdui._layout_notify(hLayout, nEvent, wParam, lParam)

    def layoutSetChildProp(self, hLayout: int, hObj: int, dwPropID: int, pvValue: int) -> bool:
        return libexdui._layout_setchildprop(hLayout, hObj, dwPropID, pvValue)

    def layoutSetProp(self, hLayout: int, dwPropID: int, pvValue: int) -> bool:
        return libexdui._layout_setprop(hLayout, dwPropID, pvValue)

    def layoutTableSetInfo(self, hLayout: int, aRowHeight: int, cRows: int, aCellWidth: int, cCells: int) -> bool:
        libexdui._layout_table_setinfo.argtypes = (
            c_int, c_void_p, c_int, c_void_p, c_int)
        return libexdui._layout_table_setinfo(hLayout, aRowHeight, cRows, aCellWidth, cCells)

    def layoutUpdate(self, hLayout: int) -> bool:
        return libexdui._layout_update(hLayout)

    def matrixCreate(self) -> int:
        libexdui._matrix_create.restype = c_void_p
        return libexdui._matrix_create()

    def matrixDestroy(self, pMatrix: int) -> bool:
        libexdui._matrix_destroy.argtypes = (c_void_p,)
        return libexdui._matrix_destroy(pMatrix)

    def matrixReset(self, pMatrix: int) -> bool:
        libexdui._matrix_reset.argtypes = (c_void_p,)
        return libexdui._matrix_reset(pMatrix)

    def matrixRotate(self, pMatrix: int, fAngle: float) -> bool:
        libexdui._matrix_rotate.argtypes = (c_void_p, c_float)
        return libexdui._matrix_rotate(pMatrix, fAngle)

    def matrixScale(self, pMatrix: int, scaleX: float, scaleY: float) -> bool:
        libexdui._matrix_scale.argtypes = (c_void_p, c_float, c_float)
        return libexdui._matrix_scale(pMatrix, scaleX, scaleY)

    def matrixTranslate(self, pMatrix: int, offsetX: float, offsetY: float) -> bool:
        libexdui._matrix_translate.argtypes = (c_void_p, c_float, c_float)
        return libexdui._matrix_translate(pMatrix, offsetX, offsetY)

    def pathAddarc(self, hPath: int, x1: float, y1: float, x2: float, y2: float, radiusX: float, radiusY: float, fClockwise: bool) -> bool:
        libexdui._path_addarc.argtypes = (
            c_int, c_float, c_float, c_float, c_float, c_float, c_float, c_bool)
        return libexdui._path_addarc(hPath, x1, y1, x2, y2, radiusX, radiusY, fClockwise)

    def pathAddarc2(self, hPath: int, x: float, y: float, width: float, height: float, startAngle: float, sweepAngle: float) -> bool:
        libexdui._path_addarc2.argtypes = (
            c_int, c_float, c_float, c_float, c_float, c_float, c_float)
        return libexdui._path_addarc2(hPath, x, y, width, height, startAngle, sweepAngle)

    def pathAddarc3(self, hPath: int, x: float, y: float, radiusX: float, radiusY: float, startAngle: float, sweepAngle: float, fClockwise: bool, barcSize: bool) -> bool:
        libexdui._path_addarc3.argtypes = (
            c_int, c_float, c_float, c_float, c_float, c_float, c_float, c_bool, c_bool)
        return libexdui._path_addarc3(hPath, x, y, radiusX, radiusY, startAngle, sweepAngle, fClockwise, barcSize)

    def pathAddbezier(self, hPath: int, x1: float, y1: float, x2: float, y2: float, x3: float, y3: float) -> bool:
        libexdui._path_addbezier.argtypes = (
            c_int, c_float, c_float, c_float, c_float, c_float, c_float)
        return libexdui._path_addbezier(hPath, x1, y1, x2, y2, x3, y3)

    def pathAddline(self, hPath: int, x1: float, y1: float, x2: float, y2: float) -> bool:
        libexdui._path_addline.argtypes = (
            c_int, c_float, c_float, c_float, c_float)
        return libexdui._path_addline(hPath, x1, y1, x2, y2)

    def pathAddQuadraticBezier(self, hPath: int, x1: float, y1: float, x2: float, y2: float) -> bool:
        libexdui._path_addquadraticbezier.argtypes = (
            c_int, c_float, c_float, c_float, c_float)
        return libexdui._path_addquadraticbezier(hPath, x1, y1, x2, y2)

    def pathAddRect(self, hPath: int, left: float, top: float, right: float, bottom: float) -> bool:
        libexdui._path_addrect.argtypes = (
            c_int, c_float, c_float, c_float, c_float)
        return libexdui._path_addrect(hPath, left, top, right, bottom)

    def pathAddRoundedRect(self, hPath: int, left: float, top: float, right: float, bottom: float, radiusTopLeft: float, radiusTopRight: float, radiusBottomLeft: float, radiusBottomRight: float) -> bool:
        libexdui._path_addroundedrect.argtypes = (
            c_int, c_float, c_float, c_float, c_float, c_float, c_float, c_float, c_float)
        return libexdui._path_addroundedrect(hPath, left, top, right, bottom, radiusTopLeft, radiusTopRight, radiusBottomLeft, radiusBottomRight)

    def pathBeginFigure(self, hPath: int) -> bool:
        return libexdui._path_beginfigure(hPath)

    def pathBeginFigure2(self, hPath: int, x: float, y: float) -> bool:
        libexdui._path_beginfigure2.argtypes = (c_int, c_float, c_float)
        return libexdui._path_beginfigure2(hPath, x, y)

    def pathBeginFigure3(self, hPath: int, x: float, y: float, figureBegin: int) -> bool:
        libexdui._path_beginfigure3.argtypes = (c_int, c_float, c_float, c_int)
        return libexdui._path_beginfigure3(hPath, x, y, figureBegin)

    def pathClose(self, hPath: int) -> bool:
        return libexdui._path_close(hPath)

    def pathCreate(self, dwFlags: int, hPath: int) -> bool:
        libexdui._path_create.argtypes = (c_int, c_void_p)
        return libexdui._path_create(dwFlags, hPath)

    def pathDestroy(self, hPath: int) -> bool:
        return libexdui._path_destroy(hPath)

    def pathEndFigure(self, hPath: int, fCloseFigure: bool) -> bool:
        return libexdui._path_endfigure(hPath, fCloseFigure)

    def pathGetBounds(self, hPath: int, lpBounds: int) -> bool:
        libexdui._path_getbounds.argtypes = (c_int, c_void_p)
        return libexdui._path_getbounds(hPath, lpBounds)

    def pathHittest(self, hPath: int, x: float, y: float) -> bool:
        libexdui._path_hittest.argtypes = (c_int, c_float, c_float)
        return libexdui._path_hittest(hPath, x, y)

    def pathOpen(self, hPath: int) -> bool:
        return libexdui._path_open(hPath)

    def pathReset(self, hPath: int) -> bool:
        return libexdui._path_reset(hPath)

    def rgnCombine(self, hRgnSrc: int, hRgnDst: int, nCombineMode: int, dstOffsetX: int, dstOffsetY: int) -> int:
        libexdui._rgn_combine.argtypes = (
            c_void_p, c_void_p, c_int, c_int, c_int)
        libexdui._rgn_combine.restype = c_void_p
        return libexdui._rgn_combine(hRgnSrc, hRgnDst, nCombineMode, dstOffsetX, dstOffsetY)

    def rgnCreateFromPath(self, hPath: int) -> int:
        libexdui._rgn_createfrompath.restype = c_void_p
        return libexdui._rgn_createfrompath(hPath)

    def rgnCreateFromRect(self, left: float, top: float, right: float, bottom: float) -> int:
        libexdui._rgn_createfromrect.argtypes = (
            c_float, c_float, c_float, c_float)
        libexdui._rgn_createfromrect.restype = c_void_p
        return libexdui._rgn_createfromrect(left, top, right, bottom)

    def rgnCreateFromRoundRect(self, left: float, top: float, right: float, bottom: float, radiusX: float, radiusY: float) -> int:
        libexdui._rgn_createfromroundrect.argtypes = (
            c_float, c_float, c_float, c_float, c_float, c_float)
        libexdui._rgn_createfromroundrect.restype = c_void_p
        return libexdui._rgn_createfromroundrect(left, top, right, bottom, radiusX, radiusY)

    def rgnDestroy(self, hRgn: int) -> bool:
        libexdui._rgn_destroy.argtypes = (c_void_p,)
        return libexdui._rgn_destroy(hRgn)

    def rgnHittest(self, hRgn: int, x: float, y: float) -> bool:
        libexdui._rgn_hittest.argtypes = (c_void_p, c_float, c_float)
        return libexdui._rgn_hittest(hRgn, x, y)

    def allocBuffer(self, dwSize: int) -> int:
        libexdui.Ex_AllocBuffer.restype = c_void_p
        return libexdui.Ex_AllocBuffer(dwSize)

    def atom(self, lptstring: str) -> int:
        libexdui.Ex_Atom.argtypes = (c_wchar_p,)
        return libexdui.Ex_Atom(lptstring)

    def dUIBindWindow(self, hWnd: int,  dwStyle: int) -> int:
        libexdui.Ex_DUIBindWindow.argtypes = (c_size_t, c_void_p, c_int)
        return libexdui.Ex_DUIBindWindow(hWnd, 0, dwStyle)

    def duiBindWindowEx(self, hwnd: int, style: int, lparam: int, msgproc: int) -> int:
        libexdui.Ex_DUIBindWindowEx.argtypes = (
            c_size_t, c_void_p, c_int, c_longlong, c_void_p)
        return libexdui.Ex_DUIBindWindowEx(hwnd, 0, style, lparam, msgproc)

    def duiFromWindow(self, hWnd: int) -> int:
        return libexdui.Ex_DUIFromWindow(hWnd)

    def duiGetClientRect(self, hExDui: int, lpClientRect: int) -> bool:
        libexdui.Ex_DUIGetClientRect.argtypes = (c_int, c_void_p)
        return libexdui.Ex_DUIGetClientRect(hExDui, lpClientRect)

    def duiGetLong(self, hExDui: int, nIndex: int) -> int:
        return libexdui.Ex_DUIGetLong(hExDui, nIndex)

    def duiGetObjFromPoint(self, handle: int, x: int, y: int) -> int:
        return libexdui.Ex_DUIGetObjFromPoint(handle, x, y)

    def duiSetLong(self, hExDui: int, nIndex: int, dwNewlong: int) -> int:
        return libexdui.Ex_DUISetLong(hExDui, nIndex, dwNewlong)

    def duiShowWindow(self, hexdui: int) -> bool:
        return libexdui.Ex_DUIShowWindow(hexdui, 1, 0, 0)

    def duiShowWindowEx(self, hExDui: int, nCmdShow: int, dwTimer: int, dwFrames: int, dwFlags: int, uEasing: int, wParam, lParam) -> bool:
        libexdui.Ex_DUIShowWindowEx.argtypes = (
            c_int, c_int, c_int, c_int, c_int, c_int, c_size_t, c_longlong)
        return libexdui.Ex_DUIShowWindowEx(hExDui, nCmdShow, dwTimer, dwFrames, dwFlags, uEasing, wParam, lParam)

    def duiTrayIconPopup(self, hExDui: int, lpwzInfo: str, lpwzInfoTitle: str, dwInfoFlags: int) -> bool:
        libexdui.Ex_DUITrayIconPopup.argtypes = (
            c_int, c_wchar_p, c_wchar_p, c_int)
        return libexdui.Ex_DUITrayIconPopup(hExDui, lpwzInfo, lpwzInfoTitle, dwInfoFlags)

    def duiTrayIconSet(self, hExDui: int, hIcon: int, lpwzTips: str) -> bool:
        libexdui.Ex_DUITrayIconSet.argtypes = (c_int, c_size_t, c_wchar_p)
        return libexdui.Ex_DUITrayIconSet(hExDui, hIcon, lpwzTips)

    def freeBuffer(self, lpBuffer: int) -> bool:
        libexdui.Ex_FreeBuffer.argtypes = (c_void_p,)
        return libexdui.Ex_FreeBuffer(lpBuffer)

    def getLastError(self) -> int:
        return libexdui.Ex_GetLastError()

    def isDxRender(self) -> bool:
        return libexdui.Ex_IsDxRender()

    def loadImageFromMemory(self, lpData: int, dwLen: int, uType: int, nIndex: int) -> int:
        libexdui.Ex_LoadImageFromMemory.argtypes = (
            c_void_p, c_size_t, c_int, c_int)
        libexdui.Ex_LoadImageFromMemory.restype = c_void_p
        return libexdui.Ex_LoadImageFromMemory(lpData, dwLen, uType, nIndex)

    def messageBox(self, handle: int, lpText: str, lpCaption: str, uType: int, dwFlags: int) -> int:
        libexdui.Ex_MessageBox.argtypes = (
            c_int, c_wchar_p, c_wchar_p, c_int, c_int)
        return libexdui.Ex_MessageBox(handle, lpText, lpCaption, uType, dwFlags)

    def messageBoxEx(self, handle: int, lpText: str, lpCaption: str, uType: int, lpCheckBox: str, lpCheckBoxChecked: int, dwMilliseconds: int, dwFlags: int, lpfnMsgProc: int) -> int:
        libexdui.Ex_MessageBoxEx.argtypes = (
            c_int, c_wchar_p, c_wchar_p, c_int, c_wchar_p, c_void_p, c_int, c_int, c_void_p)
        return libexdui.Ex_MessageBoxEx(handle, lpText, lpCaption, uType, lpCheckBox, lpCheckBoxChecked, dwMilliseconds, dwFlags, lpfnMsgProc)

    def objBeginPaint(self, hObj: int, lpPS: int) -> bool:
        libexdui.Ex_ObjBeginPaint.argtypes = (c_int, c_void_p)
        return libexdui.Ex_ObjBeginPaint(hObj, lpPS)

    def objCallProc(self, lpPrevObjProc: int, hWnd: int, hObj: int, uMsg: int, wParam, lParam) -> int:
        libexdui.Ex_ObjCallProc.argtypes = (
            c_void_p, c_size_t, c_int, c_int, c_size_t, c_longlong)
        return libexdui.Ex_ObjCallProc(lpPrevObjProc, hWnd, hObj, uMsg, wParam, lParam)

    def objCheckDropFormat(self, hObj: int, pDataObject: int, dwFormat: int) -> bool:
        libexdui.Ex_ObjCheckDropFormat.argtypes = (c_int, c_void_p, c_int)
        return libexdui.Ex_ObjCheckDropFormat(hObj, pDataObject, dwFormat)

    def objClientToScreen(self, hObj: int, x: int, y: int) -> bool:
        libexdui.Ex_ObjClientToScreen.argtypes = (c_int, c_void_p, c_void_p)
        return libexdui.Ex_ObjClientToScreen(hObj, x, y)

    def objClientToWindow(self, hObj: int, x: int, y: int) -> bool:
        libexdui.Ex_ObjClientToWindow.argtypes = (c_int, c_void_p, c_void_p)
        return libexdui.Ex_ObjClientToWindow(hObj, x, y)

    def objCreate(self, lptszClassName: str, lptszObjTitle: str, dwStyle: int, x: int, y: int, width: int, height: int, hParent: int) -> int:
        libexdui.Ex_ObjCreate.argtypes = (
            c_wchar_p, c_wchar_p, c_int, c_int, c_int, c_int, c_int, c_int)
        return libexdui.Ex_ObjCreate(lptszClassName, lptszObjTitle, dwStyle, x, y, width, height, hParent)

    def objCreateEx(self, dwStyleEx: int, lptszClassName: str, lptszObjTitle: str, dwStyle: int, x: int, y: int, width: int, height: int, hParent: int, nID: int, dwTextFormat: int, lParam: int, lpfnMsgProc: int) -> int:
        libexdui.Ex_ObjCreateEx.argtypes = (c_int, c_wchar_p, c_wchar_p, c_int, c_int,
                                            c_int, c_int, c_int, c_int, c_int, c_int, c_longlong, c_void_p, c_void_p)
        return libexdui.Ex_ObjCreateEx(dwStyleEx, lptszClassName, lptszObjTitle, dwStyle, x, y, width, height, hParent, nID, dwTextFormat, lParam, 0, lpfnMsgProc)

    def objDefProc(self, hWnd: int, hObj: int, uMsg: int, wParam, lParam) -> int:
        libexdui.Ex_ObjDefProc.argtypes = (
            c_size_t, c_int, c_int, c_size_t, c_longlong)
        return libexdui.Ex_ObjDefProc(hWnd, hObj, uMsg, wParam, lParam)

    def objDestroy(self, hObj: int) -> bool:
        return libexdui.Ex_ObjDestroy(hObj)

    def objDisableTranslateSpaceAndEnterToClick(self, hObj: int, fDisable: bool) -> bool:
        return libexdui.Ex_ObjDisableTranslateSpaceAndEnterToClick(hObj, fDisable)

    def objDispatchMessage(self, hObj: int, uMsg: int, wParam, lParam) -> int:
        libexdui.Ex_ObjDispatchMessage.argtypes = (
            c_int, c_int, c_size_t, c_longlong)
        return libexdui.Ex_ObjDispatchMessage(hObj, uMsg, wParam, lParam)

    def objDispatchNotify(self, hObj: int, nCode: int, wParam, lParam) -> int:
        libexdui.Ex_ObjDispatchNotify.argtypes = (
            c_int, c_int, c_size_t, c_longlong)
        return libexdui.Ex_ObjDispatchNotify(hObj, nCode, wParam, lParam)

    def objDrawBackgroundProc(self, hObj: int, hCanvas: int, lprcPaint: int) -> bool:
        libexdui.Ex_ObjDrawBackgroundProc.argtypes = (c_int, c_int, c_void_p)
        return libexdui.Ex_ObjDrawBackgroundProc(hObj, hCanvas, lprcPaint)

    def objEnable(self, hObj: int, fEnable: bool) -> bool:
        return libexdui.Ex_ObjEnable(hObj, fEnable)

    def objEnableEventBubble(self, hObj: int, fEnable: bool) -> bool:
        return libexdui.Ex_ObjEnableEventBubble(hObj, fEnable)

    def objEnableIME(self, hObj: int, fEnable: bool) -> bool:
        return libexdui.Ex_ObjEnableIME(hObj, fEnable)

    def objEnablePaintingMsg(self, hObj: int, bEnable: bool) -> bool:
        return libexdui.Ex_ObjEnablePaintingMsg(hObj, bEnable)

    def objEndPaint(self, hObj: int, lpPS: int) -> bool:
        libexdui.Ex_ObjEndPaint.argtypes = (c_int, c_void_p)
        return libexdui.Ex_ObjEndPaint(hObj, lpPS)

    def objEnumChild(self, hObjParent: int, lpEnumFunc: int, lParam: int) -> bool:
        libexdui.Ex_ObjEnumChild.argtypes = (c_int, c_void_p, c_longlong)
        return libexdui.Ex_ObjEnumChild(hObjParent, lpEnumFunc, lParam)

    def objEnumProps(self, hObj: int, lpfnCbk: int, param: int) -> int:
        libexdui.Ex_ObjEnumProps.argtypes = (c_int, c_void_p, c_size_t)
        return libexdui.Ex_ObjEnumProps(hObj, lpfnCbk, param)

    def objFind(self, hObjParent: int, hObjChildAfter: int, lpClassName: str, lpTitle: str) -> int:
        libexdui.Ex_ObjFind.argtypes = (c_int, c_int, c_wchar_p, c_wchar_p)
        return libexdui.Ex_ObjFind(hObjParent, hObjChildAfter, lpClassName, lpTitle)

    def objGetBackgroundImage(self, handle: int, lpBackgroundImage: int) -> bool:
        libexdui.Ex_ObjGetBackgroundImage.argtypes = (c_int, c_void_p)
        return libexdui.Ex_ObjGetBackgroundImage(handle, lpBackgroundImage)

    def objGetClassInfo(self, hObj: int, lpClassInfo: int) -> bool:
        libexdui.Ex_ObjGetClassInfo.argtypes = (c_int, c_void_p)
        return libexdui.Ex_ObjGetClassInfo(hObj, lpClassInfo)

    def objGetClassInfoEx(self, lptszClassName: str, lpClassInfo: int) -> bool:
        libexdui.Ex_ObjGetClassInfoEx.argtypes = (c_wchar_p, c_void_p)
        return libexdui.Ex_ObjGetClassInfoEx(lptszClassName, lpClassInfo)

    def objGetClientRect(self, hObj: int, lpRect: int) -> bool:
        libexdui.Ex_ObjGetClientRect.argtypes = (c_int, c_void_p)
        return libexdui.Ex_ObjGetClientRect(hObj, lpRect)

    def objGetColor(self, hObj: int, nIndex: int) -> int:
        return libexdui.Ex_ObjGetColor(hObj, nIndex)

    def objGetDropString(self, hObj: int, pDataObject: int, lpwzBuffer: int, cchMaxLength: int) -> int:
        libexdui.Ex_ObjGetDropString.argtypes = (
            c_int, c_void_p, c_wchar_p, c_int)
        return libexdui.Ex_ObjGetDropString(hObj, pDataObject, lpwzBuffer, cchMaxLength)

    def objGetFocus(self, hExDuiOrhObj: int) -> int:
        return libexdui.Ex_ObjGetFocus(hExDuiOrhObj)

    def objGetFont(self, hObj: int) -> int:
        return libexdui.Ex_ObjGetFont(hObj)

    def objGetFromID(self, hExDuiOrhObj: int, nID: int) -> int:
        return libexdui.Ex_ObjGetFromID(hExDuiOrhObj, nID)

    def objGetFromName(self, hExDuiOrhObj: int, lpName: str) -> int:
        return libexdui.Ex_ObjGetFromName(hExDuiOrhObj, lpName)

    def objGetFromNodeID(self, hExDUIOrObj: int, nNodeID: int) -> int:
        return libexdui.Ex_ObjGetFromNodeID(hExDUIOrObj, nNodeID)

    def objGetLong(self, hObj: int, nIndex: int) -> int:
        libexdui.Ex_ObjGetLong.restype = c_longlong
        return libexdui.Ex_ObjGetLong(hObj, nIndex)

    def objGetObj(self, hObj: int, nCmd: int) -> int:
        return libexdui.Ex_ObjGetObj(hObj, nCmd)

    def objGetParent(self, hObj: int) -> int:
        return libexdui.Ex_ObjGetParent(hObj)

    def objGetParentEx(self, hObj: int, phExDUI: int) -> int:
        libexdui.Ex_ObjGetParentEx.argtypes = (c_int, c_void_p)
        return libexdui.Ex_ObjGetParentEx(hObj, phExDUI)

    def objGetProp(self, hObj: int, dwKey: int) -> int:
        return libexdui.Ex_ObjGetProp(hObj, dwKey)

    def objGetRect(self, hObj: int, lpRect: int) -> bool:
        libexdui.Ex_ObjGetRect.argtypes = (c_int, c_void_p)
        return libexdui.Ex_ObjGetRect(hObj, lpRect)

    def objGetRectEx(self, hObj: int, lpRect: int, nType: int) -> bool:
        libexdui.Ex_ObjGetRectEx.argtypes = (c_int, c_void_p, c_int)
        return libexdui.Ex_ObjGetRectEx(hObj, lpRect, nType)

    def objGetText(self, hobj: int) -> str:
        text_len = self.objGetTextLength(hobj) * 2 + 2
        ret_str = ''
        ret_str.zfill(text_len)
        ret = ctypes.c_wchar_p(ret_str)
        libexdui.Ex_ObjGetText(hobj, ret, text_len)
        return ret.value

    def objGetTextLength(self, hObj: int) -> int:
        return libexdui.Ex_ObjGetTextLength(hObj)

    def objGetTextRect(self, hObj: int, lpRect: int) -> bool:
        libexdui.Ex_ObjGetTextRect.argtypes = (c_int, c_void_p)
        return libexdui.Ex_ObjGetTextRect(hObj, lpRect)

    def objGetUIState(self, hObj: int) -> int:
        return libexdui.Ex_ObjGetUIState(hObj)

    def objHandleEvent(self, hObj: int, nEvent: int, pfnCallback: int) -> bool:
        libexdui.Ex_ObjHandleEvent.argtypes = (c_int, c_int, c_void_p)
        return libexdui.Ex_ObjHandleEvent(hObj, nEvent, pfnCallback)

    def objInitPropList(self, hObj: int, nPropCount: int) -> bool:
        return libexdui.Ex_ObjInitPropList(hObj, nPropCount)

    def objInvalidateRect(self, hObj: int, lprcRedraw: int) -> bool:
        libexdui.Ex_ObjInvalidateRect.argtypes = (c_int, c_void_p)
        return libexdui.Ex_ObjInvalidateRect(hObj, lprcRedraw)

    def objIsEnable(self, hObj: int) -> bool:
        return libexdui.Ex_ObjIsEnable(hObj)

    def objIsValidate(self, hObj: int) -> bool:
        return libexdui.Ex_ObjIsValidate(hObj)

    def objIsVisible(self, hObj: int) -> bool:
        return libexdui.Ex_ObjIsVisible(hObj)

    def objKillFocus(self, hObj: int) -> bool:
        return libexdui.Ex_ObjKillFocus(hObj)

    def objKillTimer(self, hObj: int) -> bool:
        return libexdui.Ex_ObjKillTimer(hObj)

    def objLayoutClear(self, handle: int, bChildren: bool) -> bool:
        return libexdui.Ex_ObjLayoutClear(handle, bChildren)

    def objLayoutGet(self, handle: int) -> int:
        return libexdui.Ex_ObjLayoutGet(handle)

    def objLayoutSet(self, handle: int, hLayout: int, fUpdate: bool) -> bool:
        return libexdui.Ex_ObjLayoutSet(handle, hLayout, fUpdate)

    def objLayoutUpdate(self, handle: int) -> bool:
        return libexdui.Ex_ObjLayoutUpdate(handle)

    def objMove(self, hObj: int, x: int, y: int, width: int, height: int, bRepaint: bool) -> bool:
        return libexdui.Ex_ObjMove(hObj, x, y, width, height, bRepaint)

    def objPointTransform(self, hObjSrc: int, hObjDst: int, ptX: int, ptY: int) -> bool:
        libexdui.Ex_ObjPointTransform.argtypes = (
            c_int, c_int, c_void_p, c_void_p)
        return libexdui.Ex_ObjPointTransform(hObjSrc, hObjDst, ptX, ptY)

    def objPostMessage(self, hObj: int, uMsg: int, wParam, lParam) -> bool:
        libexdui.Ex_ObjPostMessage.argtypes = (
            c_int, c_int, c_size_t, c_longlong)
        return libexdui.Ex_ObjPostMessage(hObj, uMsg, wParam, lParam)

    def objRegister(self, lptszClassName: str, dwStyle: int, dwStyleEx: int, dwTextFormat: int, cbObjExtra: int, hCursor: int, dwFlags: int, pfnObjProc: int) -> int:
        libexdui.Ex_ObjRegister.argtypes = (
            c_wchar_p, c_int, c_int, c_int, c_int, c_size_t, c_int, c_void_p)
        return libexdui.Ex_ObjRegister(lptszClassName, dwStyle, dwStyleEx, dwTextFormat, cbObjExtra, hCursor, dwFlags, pfnObjProc)

    def objRemoveProp(self, hObj: int, dwKey: int) -> int:
        return libexdui.Ex_ObjRemoveProp(hObj, dwKey)

    def objScrollEnable(self, hObj: int, wSB: int, wArrows: int) -> bool:
        return libexdui.Ex_ObjScrollEnable(hObj, wSB, wArrows)

    def objScrollGetControl(self, hObj: int, nBar: int) -> int:
        return libexdui.Ex_ObjScrollGetControl(hObj, nBar)

    def objScrollGetInfo(self, hObj: int, nBar: int, lpnMin: int, lpnMax: int, lpnPos: int, lpnTrackPos: int) -> bool:
        libexdui.Ex_ObjScrollGetInfo.argtypes = (
            c_int, c_int, c_void_p, c_void_p, c_void_p, c_void_p)
        return libexdui.Ex_ObjScrollGetInfo(hObj, nBar, lpnMin, lpnMax, lpnPos, lpnTrackPos)

    def objScrollGetPos(self, hObj: int, nBar: int) -> int:
        return libexdui.Ex_ObjScrollGetPos(hObj, nBar)

    def objScrollGetRange(self, hObj: int, nBar: int, lpnMinPos: int, lpnMaxPos: int) -> bool:
        libexdui.Ex_ObjScrollGetRange.argtypes = (
            c_int, c_int, c_void_p, c_void_p)
        return libexdui.Ex_ObjScrollGetRange(hObj, nBar, lpnMinPos, lpnMaxPos)

    def objScrollGetTrackPos(self, hObj: int, nBar: int) -> int:
        return libexdui.Ex_ObjScrollGetTrackPos(hObj, nBar)

    def objScrollSetInfo(self, hObj: int, nBar: int, Mask: int, nMin: int, nMax: int, nPage: int, nPos: int, bRedraw: bool) -> int:
        return libexdui.Ex_ObjScrollSetInfo(hObj, nBar, Mask, nMin, nMax, nPage, nPos, bRedraw)

    def objScrollSetPos(self, hObj: int, nBar: int, nPos: int, bRedraw: bool) -> int:
        return libexdui.Ex_ObjScrollSetPos(hObj, nBar, nPos, bRedraw)

    def objScrollSetRange(self, hObj: int, nBar: int, nMin: int, nMax: int, bRedraw: bool) -> int:
        return libexdui.Ex_ObjScrollSetRange(hObj, nBar, nMin, nMax, bRedraw)

    def objScrollShow(self, hObj: int, wBar: int, fShow: bool) -> bool:
        return libexdui.Ex_ObjScrollShow(hObj, wBar, fShow)

    def objSendMessage(self, hObj: int, uMsg: int, wParam, lParam) -> int:
        libexdui.Ex_ObjSendMessage.argtypes = (
            c_int, c_int, c_size_t, c_void_p)
        return libexdui.Ex_ObjSendMessage(hObj, uMsg, wParam, lParam)

    def objSetBackgroundImage(self, handle: int, lpImage: int, dwImageLen: int, x: int, y: int, dwRepeat: int,  dwFlags: int, dwAlpha: int, fUpdate: bool) -> bool:
        libexdui.Ex_ObjSetBackgroundImage.argtypes = (
            c_int, c_void_p, c_size_t, c_int, c_int, c_int, c_void_p, c_int, c_int, c_bool)
        return libexdui.Ex_ObjSetBackgroundImage(handle, lpImage, dwImageLen, x, y, dwRepeat, 0, dwFlags, dwAlpha, fUpdate)

    def objSetBackgroundPlayState(self, handle: int, fPlayFrames: bool, fResetFrame: bool, fUpdate: bool) -> bool:
        return libexdui.Ex_ObjSetBackgroundPlayState(handle, fPlayFrames, fResetFrame, fUpdate)

    def objSetBlur(self, hObj: int, fDeviation: int, bRedraw: bool) -> bool:
        return libexdui.Ex_ObjSetBlur(hObj, fDeviation, bRedraw)

    def objSetColor(self, hObj: int, nIndex: int, dwColor: int, fRedraw: bool) -> bool:
        return libexdui.Ex_ObjSetColor(hObj, nIndex, dwColor, fRedraw)

    def objEditSetSelCharFormat(self, hObj: int, dwMask: int, crText: int, wzFontFace: str, fontSize: int, yOffset: int, bBold: bool, bItalic: bool, bUnderLine: bool, bStrikeOut: bool, bLink: bool) -> int:
        libexdui.Ex_ObjEditSetSelCharFormat.argtypes = (
            c_int, c_int, c_int, c_wchar_p, c_int, c_int, c_bool, c_bool, c_bool, c_bool, c_bool)
        return libexdui.Ex_ObjEditSetSelCharFormat(hObj, dwMask, crText, wzFontFace, fontSize, yOffset, bBold, bItalic, bUnderLine, bStrikeOut, bLink)

    def objEditSetSelParFormat(self, hObj: int, dwMask: int, wNumbering: int, dxStartIndent: int, dxRightIndent: int, dxOffset: int, wAlignment: int) -> int:
        libexdui.Ex_ObjEditSetSelParFormat.argtypes = (
            c_int, c_int, c_short, c_int, c_int, c_int, c_int)
        return libexdui.Ex_ObjEditSetSelParFormat(hObj, dwMask, wNumbering, dxStartIndent, dxRightIndent, dxOffset, wAlignment)

    def objSetFocus(self, hObj: int) -> bool:
        return libexdui.Ex_ObjSetFocus(hObj)

    def objSetFont(self, hObj: int, hFont: int, fRedraw: bool) -> bool:
        return libexdui.Ex_ObjSetFont(hObj, hFont, fRedraw)

    def objSetFontFromFamily(self, hObj: int, lpszFontfamily: str, dwFontsize: int, dwFontstyle: int, fRedraw: bool) -> bool:
        libexdui.Ex_ObjSetFontFromFamily.argtypes = (
            c_int, c_wchar_p, c_int, c_int, c_bool)
        return libexdui.Ex_ObjSetFontFromFamily(hObj, lpszFontfamily, dwFontsize, dwFontstyle, fRedraw)

    def objSetIMEState(self, hObjOrExDui: int, fOpen: bool) -> bool:
        return libexdui.Ex_ObjSetIMEState(hObjOrExDui, fOpen)

    def objSetLong(self, hObj: int, nIndex: int, dwNewLong: int) -> int:
        return libexdui.Ex_ObjSetLong(hObj, nIndex, dwNewLong)

    def objSetPadding(self, hObj: int, nPaddingType: int, left: int, top: int, right: int, bottom: int, fRedraw: bool) -> bool:
        return libexdui.Ex_ObjSetPadding(hObj, nPaddingType, left, top, right, bottom, fRedraw)

    def objSetParent(self, hObj: int, hParent: int) -> bool:
        return libexdui.Ex_ObjSetParent(hObj, hParent)

    def objSetPos(self, hObj: int, hObjInsertAfter: int, x: int, y: int, width: int, height: int, flags: int) -> bool:
        return libexdui.Ex_ObjSetPos(hObj, hObjInsertAfter, x, y, width, height, flags)

    def objSetProp(self, hObj: int, dwKey: int, dwValue: int) -> int:
        return libexdui.Ex_ObjSetProp(hObj, dwKey, dwValue)

    def objSetRadius(self, hObj: int, topleft: float, topright: float, bottomright: float, bottomleft: float, fUpdate: bool) -> bool:
        libexdui.Ex_ObjSetRadius.argtypes = (
            c_int, c_float, c_float, c_float, c_float, c_bool)
        return libexdui.Ex_ObjSetRadius(hObj, topleft, topright, bottomright, bottomleft, fUpdate)

    def objSetRedraw(self, hObj: int, fCanbeRedraw: bool) -> bool:
        return libexdui.Ex_ObjSetRedraw(hObj, fCanbeRedraw)

    def objSetText(self, hObj: int, lpString: str, fRedraw: bool) -> bool:
        return libexdui.Ex_ObjSetText(hObj, lpString, fRedraw)

    def objSetTextFormat(self, hObj: int, dwTextFormat: int, bRedraw: bool) -> bool:
        return libexdui.Ex_ObjSetTextFormat(hObj, dwTextFormat, bRedraw)

    def objSetTimer(self, hObj: int, uElapse: int) -> int:
        return libexdui.Ex_ObjSetTimer(hObj, uElapse)

    def objSetUIState(self, hObj: int, dwState: int, fRemove: bool, lprcRedraw: int, fRedraw: bool) -> bool:
        libexdui.Ex_ObjSetUIState.argtypes = (
            c_int, c_int, c_bool, c_void_p, c_bool)
        return libexdui.Ex_ObjSetUIState(hObj, dwState, fRemove, lprcRedraw, fRedraw)

    def objShow(self, hObj: int, fShow: bool) -> bool:
        return libexdui.Ex_ObjShow(hObj, fShow)

    def objTooltipsPop(self, hObj: int, lpText: str) -> bool:
        return libexdui.Ex_ObjTooltipsPop(hObj, lpText)

    def objTooltipsPopEx(self, hObj: int, lpTitle: str, lpText: str, x: int, y: int, dwTime: int, nIcon: int, fShow: bool) -> bool:
        return libexdui.Ex_ObjTooltipsPopEx(hObj, lpTitle, lpText, x, y, dwTime, nIcon, fShow)

    def objTooltipsSetText(self, hObj: int, lpString: str) -> bool:
        return libexdui.Ex_ObjTooltipsSetText(hObj, lpString)

    def objUpdate(self, hObj: int) -> bool:
        return libexdui.Ex_ObjUpdate(hObj)

    # def readFile(self, filePath: str, retData: int) -> bool:
    #     return libexdui.Ex_ReadFile(filePath, retData)

    # def readResSource(self, lpname: int, lpType: str, retData: int) -> int:
    #     return libexdui.Ex_ReadResSource(lpname, lpType, retData)

    def resFree(self, hRes: int) -> None:
        libexdui.Ex_ResFree.argtypes = (c_void_p,)
        libexdui.Ex_ResFree(hRes)

    def resGetFile(self, hRes: int, lpwzPath: str, lpFile: int, dwFileLen: int) -> bool:
        libexdui.Ex_ResGetFile.argtypes = (
            c_void_p, c_wchar_p, c_void_p, c_void_p)
        return libexdui.Ex_ResGetFile(hRes, lpwzPath, lpFile, dwFileLen)

    def resGetFileFromAtom(self, hRes: int, atomPath: int, lpFile: int, dwFileLen: int) -> bool:
        libexdui.Ex_ResGetFileFromAtom.argtypes = (
            c_void_p, c_int, c_void_p, c_void_p)
        return libexdui.Ex_ResGetFileFromAtom(hRes, atomPath, lpFile, dwFileLen)

    def resLoadFromFile(self, lptszFile: str) -> int:
        libexdui.Ex_ResLoadFromFile.restype = c_void_p
        return libexdui.Ex_ResLoadFromFile(lptszFile)

    def resLoadFromMemory(self, lpData: int, dwDataLen: int) -> int:
        libexdui.Ex_ResLoadFromMemory.restype = c_void_p
        return libexdui.Ex_ResLoadFromMemory(lpData, dwDataLen)

    def scale(self, n: float) -> float:
        libexdui.Ex_Scale.argtypes = (c_float,)
        libexdui.Ex_Scale.restype = c_float
        return libexdui.Ex_Scale(n)

    def setLastError(self, nError: int) -> None:
        libexdui.Ex_SetLastError(nError)

    def sleep(self, us: int) -> None:
        libexdui.Ex_Sleep(us)

    def themeFree(self, hTheme: int) -> bool:
        libexdui.Ex_ThemeFree.argtypes = (c_void_p,)
        return libexdui.Ex_ThemeFree(hTheme)

    def themeGetColor(self, hTheme: int, nIndex: int) -> int:
        libexdui.Ex_ThemeGetColor.argtypes = (c_void_p, c_int)
        return libexdui.Ex_ThemeGetColor(hTheme, nIndex)

    def themeGetValuePtr(self, hTheme: int, atomClass: int, atomProp: int) -> int:
        libexdui.Ex_ThemeGetValuePtr.argtypes = (c_void_p, c_int, c_int)
        libexdui.Ex_ThemeGetValuePtr.restype = c_void_p
        return libexdui.Ex_ThemeGetValuePtr(hTheme, atomClass, atomProp)

    def themeLoadFromFile(self, lptszFile: str, lpKey: int, dwKeyLen: int, bDefault: bool) -> int:
        libexdui.Ex_ThemeLoadFromFile.argtypes = (
            c_wchar_p, c_void_p, c_size_t, c_bool)
        libexdui.Ex_ThemeLoadFromFile.restype = c_void_p
        return libexdui.Ex_ThemeLoadFromFile(lptszFile, lpKey, dwKeyLen, bDefault)

    def themeLoadFromMemory(self, lpData: int, dwDataLen: int, lpKey: int, dwKeyLen: int, bDefault: int) -> int:
        libexdui.Ex_ThemeLoadFromMemory.argtypes = (
            c_void_p, c_size_t, c_void_p, c_size_t, c_bool)
        libexdui.Ex_ThemeLoadFromMemory.restype = c_void_p
        return libexdui.Ex_ThemeLoadFromMemory(lpData, dwDataLen, lpKey, dwKeyLen, bDefault)

    def trackPopupMenu(self, hMenu: int, uFlags: int, x: int, y: int, nReserved: int, handle: int, lpRC: int, pfnCallback: int, dwFlags: int) -> bool:
        libexdui.Ex_TrackPopupMenu.argtypes = (
            c_size_t, c_int, c_int, c_int, c_size_t, c_int, c_void_p, c_void_p, c_int)
        return libexdui.Ex_TrackPopupMenu(hMenu, uFlags, x, y, nReserved, handle, lpRC, pfnCallback, dwFlags)

    def unInit(self) -> None:
        libexdui.Ex_UnInit()

    def wndCenterFrom(self, hWnd: int, hWndFrom: int, bFullScreen: bool) -> None:
        libexdui.Ex_WndCenterFrom(hWnd, hWndFrom, bFullScreen)

    def wndCreate(self, parent: int, title: str, left: int, top: int, width: int, height: int) -> int:
        return libexdui.Ex_WndCreate(parent, 0, title, left, top, width, height, 0, 0)

    def wndMsgLoop(self) -> int:
        return libexdui.Ex_WndMsgLoop()

    def wndRegisterClass(self, lpwzClassName: str, hIcon: int, hIconsm: int, hCursor: int) -> int:
        libexdui.Ex_WndRegisterClass.argtypes = (
            c_wchar_p, c_size_t, c_size_t, c_size_t)
        libexdui.Ex_WndRegisterClass.restype = c_short
        return libexdui.Ex_WndRegisterClass(lpwzClassName, hIcon, hIconsm, hCursor)

    def castPaintStruct(self, point: int) -> EX_PAINTSTRUCT:
        return ctypes.cast(point, ctypes.POINTER(EX_PAINTSTRUCT)).contents

    def getRed(self, argb: int) -> int:
        return lobyte(argb)

    def getGreen(self, argb) -> int:
        return lobyte(argb >> 8)

    def getBlue(self, argb: int) -> int:
        return lobyte(argb >> 16)

    def getAlpha(self, argb: int) -> int:
        return lobyte(argb >> 24)

    def RGB(self, red: int, green: int, blue: int) -> int:
        return red | green << 8 | blue << 16

    def RGBA(self, red: int, green: int, blue: int, alpha: int) -> int:
        return self.RGB(blue, green, red ) | alpha << 24

    def RGB2ARGB(self, rgb: int, alpha: int) -> int:
        return self.getRed(rgb) << 16 | self.getGreen(rgb) << 8 | self.getBlue(rgb) | alpha << 24
