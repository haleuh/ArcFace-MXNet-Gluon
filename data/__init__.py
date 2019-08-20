# https://mxnet.incubator.apache.org/versions/master/faq/recordio.html
import numpy as np


def write_list(path_out, image_list, mode='w'):
    """Helper function to write image list into the file.
    The format is as below,
    integer_image_index \t float_label_index \t path_to_image
    Note that the blank between number and tab is only used for readability.
    Parameters
    ----------
    path_out: string
    image_list: generator
    mode: string
    """
    with open(path_out, mode=mode) as fout:
        for i, item in enumerate(image_list):
            if mode == 'w':
                line = '%d\t%d\t' % (i, item[2])
            else:
                line = '%d\t%d\t' % (item[0], item[2])
            for j in item[3:]:
                if isinstance(j, int) or isinstance(j, np.int32):
                    line += '%d\t' % j
                else:
                    line += '%.5f\t' % j
            line += '%s\n' % item[1]
            fout.write(line)


def read_list(path_in, refine_index=True):
    """Reads the .lst file and generates corresponding iterator.
    Parameters
    ----------
    path_in: string
    refine_index: bool
    Returns
    -------
    item iterator that contains information in .lst file
    if refine_index=True, item index will be line index
    """
    with open(path_in) as fin:
        idx = 0
        while True:
            line = fin.readline()
            if not line:
                break
            line = [i.strip() for i in line.strip().split('\t')]
            line_len = len(line)
            # check the data format of .lst file
            if line_len < 3:
                print('lst should have at least has three parts, but only has %s parts for %s' % (line_len, line))
                continue
            try:
                if refine_index:
                    item = [idx] + [line[-1]] + [float(i) for i in line[1:-1]]
                else:
                    item = [int(line[0])] + [line[-1]] + [float(i) for i in line[1:-1]]
                idx += 1
            except Exception as e:
                print('Parsing lst met error for %s, detail: %s' % (line, e))
                continue
            yield item
