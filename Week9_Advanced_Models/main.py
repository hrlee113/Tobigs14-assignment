from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import torch

from models import CRNN
from utils import CRNN_dataset
from tqdm import tqdm
import argparse
import os


def hyperparameters() :
    """
    argparse는 하이퍼파라미터 설정, 모델 배포 등을 위해 매우 편리한 기능을 제공합니다.
    파이썬 파일을 실행함과 동시에 사용자의 입력에 따라 변수값을 설정할 수 있게 도와줍니다.

    argparse를 공부하여, 아래에 나온 argument를 받을 수 있게 채워주세요.
    해당 변수들은 모델 구현에 사용됩니다.

    ---변수명---
    변수명에 맞춰 type, help, default value 등을 커스텀해주세요 :)
    
    또한, argparse는 숨겨진 기능이 지이이이인짜 많은데, 다양하게 사용해주시면 우수과제로 가게 됩니다 ㅎㅎ
    """
    parser = argparse.ArgumentParser() 
    parser.add_argument('-path', default='./dataset/', type=str, help='데이터셋의 위치') #positional
    parser.add_argument('-savepath', default='.best_model.pth', type=str, help='best model 저장을 위한 파일명')
    parser.add_argument('-batch_size', default=1, type=int, help='배치 사이즈') 
    parser.add_argument('-epochs', default=100, type=int, help='에폭 수')
    parser.add_argument('-optim', default='adam', type=str, help='optimizer 선택')
    parser.add_argument('-lr', default=0.01, type=float, help='learning rate'), 
    parser.add_argument('-device', default='0', type=str, help='gpu number')
    parser.add_argument('-img_width', default=100, type=int, help='입력 이미지 너비')
    parser.add_argument('-img_height', default=32, type=int, help='입력 이미지 높이')
    
    return parser.parse_args()


def main():
    args = hyperparameters()


    train_path = os.path.join(args.path, 'train')
    test_path = os.path.join(args.path, 'test')

    # gpu or cpu 설정
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu') 

    # train dataset load
    train_dataset = CRNN_dataset(path=train_path, w=args.img_width, h=args.img_height)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    # test dataset load
    test_dataset = CRNN_dataset(path=test_path, w=args.img_width, h=args.img_height)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
    

    # model 정의
    model = CRNN(args.img_height, 1, 37, 256)
 
    # loss 정의
    criterion = nn.CTCLoss()
    
    if args.optim == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr,
                            betas=(0.5, 0.999))
    elif args.optim == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr)
    else:
        assert False, "옵티마이저를 다시 입력해주세요. :("

    model = model.to(device)
    best_test_loss = 100000000
    for i in range(args.epochs):
        
        print('epochs: ', i)

        print("<----training---->")
        model.train()
        for inputs, targets in tqdm(train_dataloader):
            # inputs의 dimension을 (batch, channel, h, w)로 바꿔주세요. hint: pytorch tensor에 제공되는 함수 사용
            batch_size = inputs.size(0)
            inputs = inputs.to(device)
            target_text, target_length = targets 
            target_text, target_length = target_text.to(device), target_length.to(device)
            preds = model(inputs) # 여기를 log probability로 바꿔야할 것 같은데욥...
            preds_length = Variable(torch.IntTensor([preds.size(0)] * batch_size))

            """
            CTCLoss의 설명과 해당 로스의 input에 대해 설명해주세요.
            
            CTC = Connectionist Temporal Classification
            각각의 수평적인 위치에서 annotation을 획득한 label L을 input으로 삼는다.
            이 input은 한 문자가 여러 위치단위에 있는 경우(한 글자의 크기가 커서) annotation이 중복되어 도출될 수 있기 때문에 문제가 발생하는데
            이 때 CTC는 위치와 넓이를 무시하고, ground-truth text만을 CTC Loss function에 제공하고 잘못 중복된 annotation을 제거해준다.
            그리고 이 때 생성되는 가능한 모든 gt text의 점수들의 합에 -log를 취한 값이 CTC Loss이다.
            """

            loss = criterion(preds, target_text, preds_length, target_length) / batch_size 
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        

        print("<----evaluation---->")

        """
        model.train(), model.eval()의 차이에 대해 설명해주세요.
        .eval()을 하는 이유가 무엇일까요?
        batchnorm과 dropout이 있는 모델은 train할 때와 evaluate할 때 모델이 달라지기 때문에 설정하는 것이다.
        (평가 모델에 batchnorm과 dropout을 실행한다.)
        """

        model.eval() 
        loss = 0.0

        for inputs, targets in tqdm(test_dataloader):
            with torch.no_grad():
                batch_size = inputs.size(0)
                inputs = inputs.to(device)
                target_text, target_length = targets
                target_text, target_length = target_text.to(device), target_length.to(device) # 설정한 device (gpu or cpu) 에 저장되도록
                preds = model(inputs)
                preds_length = Variable(torch.IntTensor([preds.size(0)] * batch_size))
                loss = criterion(preds, target_text, preds_length, target_length) / batch_size # test를 어떻게 할까?? 

        
        print("test loss: ", loss)
        if loss < best_test_loss:
            # loss가 bset_test_loss보다 작다면 지금의 loss가 best loss가 되겠죠?
            best_test_loss = loss
            # args.savepath을 이용하여 best model 저장하기
            PATH = args.savepath
            torch.save(model, PATH)
            print("best model 저장 성공")



if __name__=="__main__":
    main()
