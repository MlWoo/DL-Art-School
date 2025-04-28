import io
from typing import Any, Dict, List, Optional, Tuple, Union

import librosa
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import PIL
import torch
from matplotlib import cm
from torchvision.transforms import ToTensor

# matplotlib.rcParams['figure.dpi'] = 300
matplotlib.use("Agg")
plt.rcParams["font.family"] = "monospace"

COLORS = [None, "tomato", "darkviolet", "blue", "green", "red", "cyan", "magenta", "yellow", "black", "white"]
CMAPS = ["plasma", "inferno", "magma", "cividis"]


def split_text_line(text, category=1, max_words=25):
    """
    A function that splits any string based on specific character
    (returning it with the string), with maximum number of words on it
    """
    category_len = len(text) // category
    texts = [text[i : i + category_len] for i in range(0, len(text), category_len)]
    max_chars = max([max([len(part) for part in text.split(" ")]) for text in texts]) + 1
    history_pos = [
        0,
    ]
    end = False
    while not end:
        for i in range(max_words, max_words - max_chars, -1):
            pos = history_pos[-1] + i
            if pos < category_len:
                split_ok = True
                for text in texts:
                    if text[pos] != " ":
                        split_ok = False
                        break
                if split_ok:
                    history_pos.append(pos)
                    break
            else:
                history_pos.append(category_len + 1)
                end = True
                break

    split_texts = []
    for i in range(1, len(history_pos)):
        for text in texts:
            split_texts.append(text[history_pos[i - 1] : history_pos[i]])
    return "\n".join(split_texts)


def get_shape_2D(shapes_keys, lengths_dict, batch_size, max_len=12000):
    shapes = []
    for shape_keyword in shapes_keys:
        if shape_keyword is None or shape_keyword not in lengths_dict.keys():
            shape_tensor = torch.LongTensor([-1] * batch_size)
            shapes.append(shape_tensor)
        else:
            shapes.append(lengths_dict[shape_keyword])
    # defualt B T C Dataset
    if shapes is None or len(shapes) == 0:
        shapes = [torch.LongTensor([max_len] * batch_size), torch.LongTensor([-1] * batch_size)]
    return shapes


def plot_wave(
    waves,
    labels=None,
    title=None,
    split_title=False,
    sr=24000,
    width=16,
    height=3,
    align_direction="vertical",
    max_len=None,
    multimedia_keep=3,
):
    """[summary]
    Args:
        waves (list): wave list
        sr (int, optional): sample rate
        width (int, optional): width of per images. Defaults to 16.
        height (int, optional): height of per images. Defaults to 3.
        align_direction (str, optional): alignments of multiple images,
            'vertical', 'horizonal'. Defaults to 'vertical'.

    Returns:
        [Tensor]: images of each wave
    """
    if isinstance(waves, (list, tuple)):
        num_items = len(waves)
    else:
        num_items = 1

    buffer = io.BytesIO()
    images = []
    for index in range(num_items):
        waves_save = []
        for wave in waves:
            if max_len is not None:
                waves_save.append(wave[index, :max_len])
            else:
                waves_save.append(wave[index])
        if split_title:
            title = split_text_line(title)
        if align_direction == "vertical":
            figsize = (width, height * num_items)
            plt.figure(figsize=figsize)
            for i in range(num_items):
                plt.subplot(num_items, 1, i)
                librosa.display.waveplot(waves[i], sr=sr)
        elif align_direction == "horizonal":
            figsize = (width * num_items, height)
            plt.figure(figsize=figsize)
            for i in range(num_items):
                plt.subplot(num_items, i, 1)
                librosa.display.waveplot(waves[i], sr=sr)
        plt.tight_layout()
        plt.savefig(buffer, format="png")
        buffer.seek(0)
        pil_image = PIL.Image.open(buffer)
        image = ToTensor()(pil_image)
        plt.clf()
        plt.close()
        images.append(image)
    images = torch.stack(images)
    return images


def plot_core(
    data,
    fig,
    ax,
    visual_method: str,
    color: str,
    t_label: str,
    l_label: str,
    w_id: int = 0,
    h_id: int = 0,
    figures_w: int = 1,
    figures_h: int = 1,
    xlims: Optional[Tuple[float, float]] = None,
    ylims: Optional[Tuple[float, float]] = None,
    y_pos: str = "left",
    vmin: float = 0.0,
    vmax: float = 1.0,
    colorbar: bool = False,
):
    if visual_method == "show":
        im = ax.imshow(data, cmap=cm.get_cmap(color), aspect="auto", vmin=vmin, vmax=vmax, interpolation="none")
        ax.set_ylim(0, data.shape[0])
        if h_id == 0 and t_label is not None:
            ax.set_title(t_label, fontsize="medium")
        if w_id == 0 and l_label is not None:
            ax.set_ylabel(l_label, fontsize="medium")
            ax.yaxis.set_label_position("left")
            ax.tick_params(labelsize="x-small", left=True, labelleft=True)
        if colorbar and w_id == (figures_w - 1):
            # 在右侧添加颜色条
            cax = ax.inset_axes([1.02, 0.1, 0.2, 0.8])
            # 隐藏 inset_axes 的边界和坐标轴
            cax.spines["top"].set_visible(False)
            cax.spines["right"].set_visible(False)
            cax.spines["left"].set_visible(False)
            cax.spines["bottom"].set_visible(False)

            cax.set_xticks([])  # 隐藏 x 轴刻度
            cax.set_yticks([])  # 隐藏 y 轴刻度

            cbar = fig.colorbar(mappable=im, location="left", orientation="vertical", ax=cax)
            cbar.ax.yaxis.set_ticks_position("right")

    elif visual_method == "plot":
        ax.plot(data, color=color)
        ax.set_xlim(xlims[0], xlims[1])
        ax.set_ylim(ylims[0], ylims[1])
        if y_pos == "left":
            if w_id == 0:
                if l_label is not None:
                    ax.set_ylabel(l_label, color=color)
                    ax.yaxis.set_label_position("left")
                ax.tick_params(labelsize="x-small", colors=color, bottom=False, labelbottom=False)
        else:
            if w_id == (figures_w - 1):
                if l_label is not None:
                    ax.set_ylabel(l_label, color=color)
                    ax.yaxis.set_label_position("right")
                ax.tick_params(
                    labelsize="x-small",
                    colors=color,
                    bottom=False,
                    labelbottom=False,
                    left=False,
                    labelleft=False,
                    right=True,
                    labelright=True,
                )


def add_axis(fig, old_ax):
    ax = fig.add_axes(old_ax.get_position(), anchor="C")
    ax.set_facecolor("None")
    return ax


def check_cfg_list_n1(cfg: Optional[List[Any]], allow_none=True):
    # sanity_check
    if cfg is None:
        if not allow_none:
            raise ValueError("cfg should be not None")
    elif not isinstance(cfg, list):
        raise ValueError("cfg should be list")


def check_cfg_list_n1_or_n2(cfg: Union[List[Any], List[List[Any]]], allow_none=True):
    # sanity_check
    if isinstance(cfg, list):
        if isinstance(cfg[0], list):
            for cfg_ in cfg:
                if not isinstance(cfg_, list):
                    raise ValueError("list of list of cfg should be consistent")
    else:
        if not (allow_none and cfg is None):
            raise ValueError("visual_methods should be list")


def cvt_cfg_list_n2(
    cfg: Optional[Union[List[Any], List[List[Any]]]], idx: int, group_size: int, num_group: int, allow_none=True
):
    if cfg is None:
        if allow_none:
            group_cfg = [None] * group_size
        else:
            raise ValueError("cfg should be not None")
    elif isinstance(cfg[0], list):
        if len(cfg[idx]) != group_size:
            raise ValueError("list of list of tensors and cfg should be consistent")
        group_cfg = cfg[idx]
    else:
        if group_size != num_group:
            raise ValueError("list of list of tensors and cfg should be consistent")
        group_cfg = cfg
    return group_cfg


def cvt_cfg_list_n1(cfg: Optional[Union[List[Any], List[List[Any]]]], group_size: int, allow_none=True):
    if cfg is None:
        if allow_none:
            group_cfg = [None] * group_size
        else:
            raise ValueError("cfg should be not None")
    elif isinstance(cfg[0], list) or len(cfg) != group_size:
        raise ValueError("list of tensors and methods should be consistent")
    else:
        group_cfg = cfg
    return [group_cfg]


def cvt_cfg_list(cfg: Optional[List[Any]], group_size: int, allow_none=True):
    if cfg is None:
        if allow_none:
            group_cfg = [None] * group_size
        else:
            raise ValueError("cfg should be not None")
    elif isinstance(cfg, list) or len(cfg) != group_size:
        raise ValueError("list of tensors and cfg should be consistent")
    else:
        group_cfg = cfg
    return group_cfg


def plot_images(
    log_vars: Dict[str, Any],
    tensor_keys: Union[List[str], List[List[str]]],
    visual_methods: Union[List[str], List[List[str]]] = [
        "show",
    ],
    color_info: Optional[Union[List[str], List[List[str]]]] = None,
    y_lim_info: Optional[Union[List[Tuple[float, float]], List[List[Tuple[float, float]]]]] = None,
    y_pos_info: Optional[Union[List[str], List[List[str]]]] = None,
    t_labels: Optional[Union[List[str], List[List[str]]]] = None,
    l_labels: Optional[Union[List[str], List[List[str]]]] = None,
    indice: Optional[List[int]] = None,
    titles: Optional[List[str]] = None,
    shapes_keys: Optional[List[Tuple[str, str]]] = None,
    texts=None,
    split_text=False,
    num_split=2,
    max_words=80,
    width: int = 0,
    height: int = 0,
    align_direction: str = "v",
    colorbar: bool = False,
):
    text_figure_num = 0 if texts is None else 1

    # sanity_check
    check_cfg_list_n1_or_n2(visual_methods, False)
    for cfg in [color_info, y_lim_info, y_pos_info, t_labels, l_labels]:
        check_cfg_list_n1_or_n2(cfg, True)

    for cfg in [indice, titles, shapes_keys]:
        check_cfg_list_n1(cfg, True)

    if isinstance(tensor_keys, list):
        if isinstance(tensor_keys[0], list):
            num_group = len(tensor_keys)
            group_size = len(tensor_keys[0])
            group_methods = []
            group_y_lim_info = []
            group_y_pos_info = []
            group_color_info = []
            group_t_labels = []
            group_l_labels = []
            for i, tensor_list in enumerate(tensor_keys):
                group_size_ = len(tensor_list)
                if isinstance(tensor_list, list):
                    group_methods.append(cvt_cfg_list_n2(visual_methods, i, group_size_, group_size, False))
                    group_y_lim_info.append(cvt_cfg_list_n2(y_lim_info, i, group_size_, group_size, True))
                    group_y_pos_info.append(cvt_cfg_list_n2(y_pos_info, i, group_size_, group_size, True))
                    group_color_info.append(cvt_cfg_list_n2(color_info, i, group_size_, group_size, True))
                    group_t_labels.append(cvt_cfg_list_n2(t_labels, i, group_size_, group_size, True))
                    group_l_labels.append(cvt_cfg_list_n2(l_labels, i, group_size_, group_size, True))
                else:
                    raise ValueError("list of list of tensor_keys should be consistent")
            # grouping
            group_tensor_keys = tensor_keys
        else:
            num_group = 1
            group_tensor_keys = [tensor_keys]
            group_size = len(tensor_keys)
            group_methods = cvt_cfg_list_n1(visual_methods, group_size, False)
            group_y_lim_info = cvt_cfg_list_n1(y_lim_info, group_size, True)
            group_y_pos_info = cvt_cfg_list_n1(y_pos_info, group_size, True)
            group_color_info = cvt_cfg_list_n1(color_info, group_size, True)
            group_t_labels = cvt_cfg_list_n1(t_labels, group_size, True)
            group_l_labels = cvt_cfg_list_n1(l_labels, group_size, True)
    else:
        raise ValueError("data orgnization is not supported")

    # anchor_tensor
    # B nH nW H W
    # B nH/nW H W
    # B H W
    if align_direction == "v":
        num_group_h = num_group
        num_group_w = 1
    else:
        num_group_h = 1
        num_group_w = num_group

    figures_h = 1
    figures_w = 1
    valid_data_flag = False
    for tensor_keys_, methods_ in zip(group_tensor_keys, group_methods):
        for tensor_key, method in zip(tensor_keys_, methods_):
            if tensor_key in log_vars:
                anchor_2d = method == "show"
                anchor_tensor = log_vars[tensor_key]
                valid_data_flag = True
                break
        if valid_data_flag:
            break
    if not valid_data_flag:
        return None

    ndim_ = anchor_tensor.dim()
    ndim = ndim_ if anchor_2d else ndim_ + 1

    if ndim > 5 or ndim < 3:
        raise ValueError("Could not plot the tensor_keys")
    if ndim == 5:
        figures_h, figures_w = anchor_tensor.shape[1], anchor_tensor.shape[2]
    elif ndim == 4:
        if align_direction == "v" and num_group == 1:
            figures_h = anchor_tensor.shape[1]
        else:
            figures_w = anchor_tensor.shape[1]

    group_titles = cvt_cfg_list(titles, num_group, allow_none=True)
    if shapes_keys is None:
        group_shape_keys = [
            (None, None),
        ] * num_group
    elif isinstance(shapes_keys, list):
        if len(shapes_keys) == 1:
            group_shape_keys = shapes_keys * num_group
        elif len(shapes_keys) != num_group:
            raise ValueError("list of tensors and methods should be consistent")
        else:
            group_shape_keys = shapes_keys
    group_shapes = [get_shape_2D(shape_keys, log_vars, anchor_tensor.shape[0]) for shape_keys in group_shape_keys]
    buffer = io.BytesIO()
    if indice is None:
        if ndim == 5:
            indice = list(range(1))
        else:
            indice = list(range(anchor_tensor.shape[0]))

    images = []
    for index in indice:
        fig = plt.figure(
            figsize=(num_group_w * figures_w * width, (num_group_h * figures_h + text_figure_num) * height)
        )
        for i, (
            tensor_keys_,
            methods_,
            y_lim_info_,
            y_pos_info_,
            color_info_,
            title,
            t_labels_,
            l_labels_,
            shapes,
        ) in enumerate(
            zip(
                group_tensor_keys,
                group_methods,
                group_y_lim_info,
                group_y_pos_info,
                group_color_info,
                group_titles,
                group_t_labels,
                group_l_labels,
                group_shapes,
            )
        ):
            # only one show in a images
            show_count = sum([1 if method == "show" else 0 for method in methods_])
            if show_count > 1:
                raise ValueError("only one show method in a same image to avoid overcover")
            # layout share the same shapes
            for j in range(figures_h):
                for k in range(figures_w):
                    ax_o = fig.add_subplot(
                        num_group_h * figures_h + text_figure_num,
                        num_group_w * figures_w,
                        i * figures_w * figures_h + j * figures_w + k + 1,
                    )
                    for m, (tensor_key, method, y_lims, y_pos, color, t_label, l_label) in enumerate(
                        zip(tensor_keys_, methods_, y_lim_info_, y_pos_info_, color_info_, t_labels_, l_labels_)
                    ):
                        if tensor_key in log_vars:
                            tensor = log_vars[tensor_key]
                        else:
                            continue
                        tensor_plane = tensor[index].float().cpu().numpy().astype(np.float32)
                        ndim_ = tensor_plane.ndim
                        if ndim_ == 4:
                            if method != "show":
                                raise ValueError("2D data should use show method")
                            data = tensor_plane[:, :, : shapes[1][index].item(), : shapes[0][index].item()]
                            i_data = data[j, k]
                        elif ndim_ == 3:
                            if method == "plot":
                                data = tensor_plane[:, :, : shapes[1][index].item()]
                                i_data = data[j, k]
                            else:
                                data = tensor_plane[:, : shapes[1][index].item(), : shapes[0][index].item()]
                                i_data = data[j + k]
                        elif ndim_ == 2:
                            if method == "plot":
                                data = tensor_plane[:, : shapes[0][index].item()]
                                i_data = data[j + k]
                            else:
                                data = tensor_plane[: shapes[1][index].item(), : shapes[0][index].item()]
                                i_data = data
                        elif ndim_ == 1:
                            if method != "plot":
                                raise ValueError("1D data should use plot method")
                            data = tensor_plane[: shapes[1][index].item()]
                            i_data = data
                        else:
                            raise ValueError(f"illegal value ndim {ndim_}")

                        if method == "show":
                            vmax = np.max(data)
                            vmin = np.min(data)
                        else:
                            vmax = 1.0
                            vmin = 0.0
                        if m == 0:
                            ax = ax_o
                            xlims = (0, i_data.shape[-1])
                        else:
                            ax = add_axis(fig, ax_o)
                        if figures_h > 1:
                            l_label = l_label + f" {j}"
                        if figures_w > 1:
                            t_label = t_label + f" {k}"
                        plot_core(
                            i_data,
                            fig,
                            ax,
                            method,
                            color,
                            t_label,
                            l_label,
                            k,
                            j,
                            figures_w,
                            figures_h,
                            xlims,
                            y_lims,
                            y_pos,
                            vmax=vmax,
                            vmin=vmin,
                            colorbar=colorbar,
                        )
        if texts is not None:
            text = texts[index]
            if isinstance(text, str):
                if split_text:
                    text = split_text_line(
                        text, category=num_split, max_words=int(max_words * num_group_w * figures_w * 0.9)
                    )
                # Set common labels
                fig.text(
                    0.1 / (num_group_w * figures_w),
                    0.2 / (num_group_h * figures_h + text_figure_num),
                    text,
                    ha="left",
                    fontsize=12,
                )
            elif isinstance(text, torch.Tensor):
                # remove white space
                left, right = 0, text.shape[1] - 1
                white_col = torch.ones(text.shape[0], 3)
                while torch.equal(text[:, left], white_col):
                    left += 1
                while torch.equal(text[:, right], white_col):
                    right -= 1
                ax_o = fig.add_subplot(
                    num_group_h * figures_h + text_figure_num, 1, num_group_h * figures_h + text_figure_num
                )
                ax_o.imshow(text[:, left : right + 1])
                ax_o.axis("off")
        # plt.tight_layout()
        buffer.seek(0)
        plt.savefig(buffer, format="jpg", bbox_inches="tight", dpi=200)
        buffer.seek(0)
        image = PIL.Image.open(buffer)
        image = ToTensor()(image)
        plt.close()
        images.append(image)
    images_tensor = torch.stack(images)
    return images_tensor


def plot_table(
    valss: list, colnamess: list, rownamess: list, a_col_width: int = 2, a_row_height: int = 3, fontsize: int = 10
):
    leg = max([len(vals[0]) for vals in valss])
    cnt = len(valss)
    _, axes = plt.subplots(cnt, 1, figsize=(leg * a_col_width, cnt * a_row_height))
    buffer = io.BytesIO()

    for i, (vals, colnames, rownames) in enumerate(zip(valss, colnamess, rownamess)):
        colors = [["white"] * len(colnames)] * len(rownames)
        for ic in range(len(colnames)):
            if vals[3][ic]:
                for ir in range(len(rownames)):
                    colors[ir][ic] = "lightsteelblue"
        if cnt > 1:
            ax = axes[i]
        else:
            ax = axes
        tab = ax.table(
            cellText=vals,
            colLabels=colnames,
            rowLabels=rownames,
            loc="center",
            cellLoc="center",
            rowLoc="center",
            cellColours=colors,
        )
        tab.auto_set_font_size(False)
        tab.set_fontsize(fontsize)
        tab.auto_set_column_width(col=list(range(len(colnames))))
        ax.axis("tight")
        ax.axis("off")
        ax.set_title(f"Sample#{i}")
    buffer.seek(0)
    plt.savefig(buffer, format="jpg")
    buffer.seek(0)
    image = PIL.Image.open(buffer)
    image = ToTensor()(image)
    plt.close()
    image_tensor = torch.stack([image])
    return image_tensor
