import mindspore

import numpy as np

from mindone.metrics.inception import InceptionScore

EXPECTED_SCORE = 1.0538204
EXPECTED_SCORE_STD = 0.00949103
ACCEPTED_ERROR = 0.001

def inception_score_metric():
    imgs = mindspore.Tensor(
        np.loadtxt('D:\\__AREA_WORKING__\\codes\\playground\\playground\\tensor.txt')
    )
    imgs = imgs.reshape((100, 3, 299, 299)).to(mindspore.uint8)
    inception = InceptionScore(deactivate_randperm=True)
    inception.update(imgs)
    inception_score = inception.eval()
    score_error = np.abs(inception_score[0].asnumpy() - EXPECTED_SCORE)
    print(inception_score[0])
    std_error = np.abs(inception_score[1].asnumpy() - EXPECTED_SCORE_STD)
    print(inception_score[1])
    assert score_error < ACCEPTED_ERROR
    assert std_error < ACCEPTED_ERROR

if __name__ == "__main__":
    inception_score_metric()