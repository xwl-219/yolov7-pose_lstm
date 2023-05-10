# yolov7-pose_lstm

            +-----------------+
            |       LSTM      |
            |       (64)      |
            |  return_sequences=True |
            +--------+--------+
                     |
            +--------+--------+
            | LayerNormalization |
            |    (axis=1)   |
            +--------+--------+
                     |
            +--------+--------+
            |       LSTM      |
            |      (128)      |
            |  return_sequences=True |
            +--------+--------+
                     |
            +--------+--------+
            |       LSTM      |
            |      (128)      |
            |  return_sequences=True |
            +--------+--------+
                     |
            +--------+--------+
            | LayerNormalization |
            |    (axis=1)   |
            +--------+--------+
                     |
            +--------+--------+
            |       LSTM      |
            |       (64)      |
            |  return_sequences=False |
            +--------+--------+
                     |
            +--------+--------+
            |      Dense      |
            |      (64)       |
            |  activation='relu' |
            +--------+--------+
                     |
            +--------+--------+
            |      Dense      |
            |      (32)       |
            |  activation='relu'|
            +--------+--------+
                     |
            +--------+--------+
            |      Dense      |
            |  (actions.shape[0],) |
            |  activation='softmax' |
            +--------+--------+