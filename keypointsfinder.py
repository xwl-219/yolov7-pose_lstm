# Importing the Required Libraries

import cv2
# To calculate the FPS
import time


#As YOLOv7 is built using PyTorch, To perform pose estimation using YOLOv7 we need to import the PyTorch module.
#To use the PyTorch library we do, import torch
# To check the version of the PyTorch library we do print(torch. __version__)

import torch

# The argparse module is used for the command line interface
import argparse

# To convert the list into the numpy array we use the numpy library
import numpy as np


from utils.datasets import letterbox
from utils.torch_utils import select_device
from models.experimental import attempt_load
from utils.plots import output_to_keypoint, plot_skeleton_kpts, plot_one_box_kpt, colors
from utils.general import non_max_suppression_kpt, strip_optimizer
from torchvision import transforms
import tensorflow
from PIL import ImageFont, ImageDraw, Image


# Creating a load_classes function so we can load the coco.names file and after reading each of 
#the name in coco.names file we can return the names in the form of a list
def load_classes(path):
    # Loads *.names file at 'path'
    with open(path, 'r') as f:
        names = f.read().split('\n')
    return list(filter(None, names)) 

# This is our Main Function
@torch.no_grad()
def run(poseweights='yolov7-w6-pose.pt', source='pose.mp4', device='cpu', names = 'utils/coco.names', line_thickness = 2):

    path = source
    ext = path.split('/')[-1].split('.')[-1].strip().lower()
    if ext in ["mp4", "webm", "avi"] or ext not in ["mp4", "webm", "avi"] and ext.isnumeric():
        input_path = int(path) if path.isnumeric() else path
        device = select_device(opt.device)
        names = load_classes(names)
        half = device.type != 'cpu'
        model = attempt_load(poseweights, map_location=device)
        _ = model.eval()

        cap = cv2.VideoCapture(input_path)
        webcam = False

        if (cap.isOpened() == False):
            print('Error while trying to read video. Please check path again')

        fw, fh = int(cap.get(3)), int(cap.get(4))
        if ext.isnumeric():
            webcam = True
            fw, fh = 1280, 768
        vid_write_image = letterbox(
            cap.read()[1], (fw), stride=64, auto=True)[0]

        resize_height, resize_width = vid_write_image.shape[:2]
        out_video_name = "output" if path.isnumeric(
        ) else f"{input_path.split('/')[-1].split('.')[0]}"
        out = cv2.VideoWriter(f"{out_video_name}_wed.mp4", cv2.VideoWriter_fourcc(
            *'mp4v'), 30, (resize_width, resize_height))
        if webcam:
            out = cv2.VideoWriter(f"{out_video_name}_mon.mp4", cv2.VideoWriter_fourcc(
                *'mp4v'), 30, (fw, fh))

        frame_count, total_fps = 0, 0

        # ====Loading the  custom font ===========
        fontpath = "sfpro.ttf"
        font = ImageFont.truetype(fontpath, 32)
        # ===================================
        sequence = []
        keypoints = []
        j = 1
        seq = 30
        # =============================================
        while cap.isOpened:

           # print(f"Frame {frame_count} Processing")
            ret, frame = cap.read()
            if ret:
                origimage = frame
                #Creating a Black Mask
                mask=np.zeros(frame.shape[:2] , dtype="uint8") 
                #Defining the ROI for the Player 2 only
                roi = cv2.rectangle(mask, (0, 0), (1920, 1080),(255,255,255), -1)
                #Overlapping the original image on the black mask
                masked=cv2.bitwise_and(frame,frame,mask=mask)
                masked[np.where((masked==[0,0,0]).all(axis=2))]=[255,0,0]
                orig_image = masked

                # preprocess image
                image_ = cv2.cvtColor(origimage, cv2.COLOR_BGR2RGB)
                image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
                image_ = letterbox(image_, (resize_width), stride=64, auto=True)[0]

                image = letterbox(image, (resize_width), stride=64, auto=True)[0]
                image__ = image_.copy()
                image___ = image.copy()
                image_ = transforms.ToTensor()(image_)
                image = transforms.ToTensor()(image)
                image_ = torch.tensor(np.array([image_.numpy()]))
                image = torch.tensor(np.array([image.numpy()]))
                image_ = image_.to(device)
                image_ = image_.float()
                image = image.to(device)
                image = image.float()
                start_time = time.time()

                with torch.no_grad():
                    output, _ = model(image)

                output_data = non_max_suppression_kpt(output, 0.25, 0.65, nc=model.yaml['nc'], nkpt=model.yaml['nkpt'], kpt_label=True)
                output = output_to_keypoint(output_data)

                img_ = image_[0].permute(1, 2, 0) * 255
                img_ = img_.cpu().numpy().astype(np.uint8)

                img_img = cv2.cvtColor(img_, cv2.COLOR_RGB2BGR)

                img = image[0].permute(1, 2, 0) * 255
                img = img.cpu().numpy().astype(np.uint8)

                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
 
                gn = torch.tensor(img.shape)[[1, 0, 1, 0]]    
                for i, pose in enumerate(output_data):  
                
                    if len(output_data):  
                        for c in pose[:, 5].unique(): 
                            n = (pose[:, 5] == c).sum()  
                            # print("No of Objects in Current Frame : {}".format(n))
                        
                        for det_index, (*xyxy, conf, cls) in enumerate(reversed(pose[:,:6])): 
                            c = int(cls) 
                            kpts = pose[det_index, 6:]
                            label = names[c]
                            plot_one_box_kpt(xyxy, img_img, label=label, color=colors(c, True), 
                                        line_thickness=opt.line_thickness, kpts=kpts, steps=3, 
                                        orig_shape=img.shape[:2])                            
                            # preprocess model input data and return the keypoints of the body =======
                            if j <= seq:
                                for idx in range(output.shape[0]):
                                    kpts = output[idx, 7:].T
                                    plot_skeleton_kpts(img_img, kpts, 3)
                                    sequence.append(kpts.tolist())
                            #The keypoints predictions from the pose estimation model is stacked as
                            #a sequence of 30 frames and each sequence contain 51 features 
                            # (X-coordinate, y-coordinate and the confidence score of 17 key points)
                            if len(sequence) == 30:
                                keypoints.append(sequence)
                                print(sequence)
                               # print(keypoints)
                            if j == seq:
                                sequence = []
                                j = 0
                            j += 1

                if webcam:
                    #cv2.imshow("Detection", img_img)
                    key = cv2.waitKey(1)
                    if key == ord('c'):
                        break
                else:
                    img_ = img.copy()
                    img_ = cv2.resize(
                        img_, (960, 540), interpolation=cv2.INTER_LINEAR)
                    #cv2.imshow("Detection", img_img)
                    cv2.waitKey(1)

                end_time = time.time()
                fps = 1 / (end_time - start_time)
                total_fps += fps
                frame_count += 1
                out.write(img_img)
            else:
                break
        print('打印keypoint：')
        print(keypoints)
        cap.release()
        avg_fps = total_fps / frame_count
        print(f"Average FPS: {avg_fps:.3f}")


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--poseweights', nargs='+', type=str,
                        default='yolov7-w6-pose.pt', help='model path(s)')
    parser.add_argument('--source', type=str,
                        help='path to video or 0 for webcam')
    parser.add_argument('--device', type=str, default='cpu',
                        help='cpu/0,1,2,3(gpu)')
    parser.add_argument('--line_thickness', default = 3, help = 'Please Input the Value of Line Thickness')

    opt = parser.parse_args()
    return opt


def main(opt):
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    strip_optimizer(opt.device, opt.poseweights)
    main(opt)
