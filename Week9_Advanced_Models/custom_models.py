# models.py 파일과 논문을 바탕으로 빈칸을 채워주세요.
import torch.nn as nn

class CRNN(nn.Module):
    def __init__(self, nh=256, nClass=37):
        super(CRNN, self).__init__()
        
        self.cnn_module = nn.Sequential(
            
            nn.Conv2d(32, 64, 3, 1, 1), # Conv1
            nn.ReLU(True),
            nn.MaxPool2d(kernel_Size=(2,2), stride=2), # MaxPool1
            nn.Conv2d(64, 128, 3, 1, 1), # Conv2
            nn.ReLU(True),
            nn.MaxPool2d(kernel_Size=(2,2), stride=2),  # MaxPool2
            nn.Conv2d(128, 256, 3, 1, 1), # Conv3
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1), # Conv4
            nn.ReLU(True),
            nn.MaxPool2d(kernel_Size=(2,1), stride=2), # MaxPool3 (kernel_size=(W,H)
            nn.Conv2d(256, 512, 3, 1, 1), # Conv5
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1), # Conv6
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_Size=(2,1), stride=2), # MaxPool4
            nn.Conv2d(512, 512, 2, 1, 0), # Conv7
            nn.ReLU(True)
            )
            
        self.rnn_model = nn.Sequential(
            nn.LSTM(512, nh, bidirectional=True),
            nn.LSTM(nh, nh, bidirectional=True)
        )


        self.embedding = nn.Linear(nh*2, nClass)  # hidden_nodes * num_of_directions , output_size

    def forward(self, input):
        conv = self.cnn_module(input)
        conv = conv.squeeze(2) #2차원 제거
        conv = conv.permute(2, 0, 1) #0->2, 1->0, 2->1 차원으로 변경
        output, _ = self.rnn_model(conv)
        seq_len, batch, h_2 =  output.size()
        output = output.view(seq_len * batch, h_2) # shape 변경
        output = self.embedding(output)
        output = output.view(seq_len, batch, -1) # shape 변경
        return output