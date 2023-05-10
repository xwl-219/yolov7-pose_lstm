#**Importing All the Required Libraries***
import cv2
import time
import torch
import argparse
import numpy as np
from utils.datasets import letterbox
from utils.torch_utils import select_device
from models.experimental import attempt_load
from utils.plots import output_to_keypoint, plot_skeleton_kpts, plot_one_box_kpt, colors
from utils.general import non_max_suppression_kpt, strip_optimizer
from torchvision import transforms
import tensorflow
from PIL import ImageFont, ImageDraw, Image

#Creating an Empty Dictionary, to save the to shot count of each type of stroke
# The first element in the dictionary is the  key, which contains the shot type 
# The second element in the dictionary is the value, which contains the shot count of each of the stroke 
# for example Smash Shot is played by the player 10 times
# while the Drop Shot happens 4 times.


object_counter = {}

#Creating a Function  by the name load_classes,using this function we will load the coco.names file and read each
#of the object name, in the coco.names file and this function will return all the object names in the coco.names
#file in the form of a list
def load_classes(path):
    # Loads *.names file at 'path'
    with open(path, 'r') as f:
        names = f.read().split('\n')
    return list(filter(None, names))  # filter removes empty strings (such as last line)

#This is the main function, here we are passing the yolov7 pose weights, by default we are setting the deivce as CPU
#setting the line thick ness of the skeleton lines as well

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

        # =3.0===Load custom font ===========
        fontpath = "sfpro.ttf"
        font = ImageFont.truetype(fontpath, 32)
        # ===================================

        # ==4.0=== Load trained pose-indentification model======
        tf_model = tensorflow.keras.models.load_model('mmodelfinal.h5')
        # ==================================================

        # == 5.0 == variable declaration===========
        sequence = []
        keypoints = []
        pose_name = ''
        posename_list = []
        actions = np.array(['Smash-Shot', 'Drop-Shot'])
        label_map = {label: num for num, label in enumerate(actions)}
        j = 1
        seq = 30
        # =============================================
        while cap.isOpened:

            print(f"Frame {frame_count} Processing")
            ret, frame = cap.read()
            if ret:
                #First, we will do for the Player 2
                origimage = frame
                #Creating a Black Mask
                mask=np.zeros(frame.shape[:2] , dtype="uint8") 
                #Setting the ROI for the Player 2
                roi = cv2.rectangle(mask, (401, 585), (1547, 1069),(255,255,255), -1)
                #Overlapping the Mask on the Original Image
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


                ###Player1 Starts From Here 
                origimage2 = frame
                #Creating a Black Mask
                mask2=np.zeros(frame.shape[:2] , dtype="uint8") 
                #Setting the Region of Interest for the Player 1
                roi = cv2.rectangle(mask2, (421, 321), (1493, 565),(255,255,255), -1)
                #Overlapping the Mask on the Original Image
                masked2=cv2.bitwise_and(frame,frame,mask=mask2)
                masked2[np.where((masked2==[0,0,0]).all(axis=2))]=[255,0,0]
                orig_image2 = masked2

                # preprocess image
                image_2 = cv2.cvtColor(origimage2, cv2.COLOR_BGR2RGB)
                image2 = cv2.cvtColor(orig_image2, cv2.COLOR_BGR2RGB)

                image_2 = letterbox(image_2, (resize_width), stride=64, auto=True)[0]

                image2 = letterbox(image2, (resize_width), stride=64, auto=True)[0]
                image__2 = image_2.copy()
                image___2 = image2.copy()
                image_2 = transforms.ToTensor()(image_2)
                image2 = transforms.ToTensor()(image2)
                image_2 = torch.tensor(np.array([image_2.numpy()]))
                image2 = torch.tensor(np.array([image2.numpy()]))
                image_2 = image_2.to(device)
                image_2 = image_2.float()
                image2 = image2.to(device)
                image2 = image2.float()
                start_time = time.time()

                with torch.no_grad():
                    output, _ = model(image)
                    output2, _ = model(image2)
                output_data = non_max_suppression_kpt(output, 0.25, 0.65, nc=model.yaml['nc'], nkpt=model.yaml['nkpt'], kpt_label=True)
                output = output_to_keypoint(output_data)
                output_data2 = non_max_suppression_kpt(output2, 0.25, 0.65, nc=model.yaml['nc'], nkpt=model.yaml['nkpt'], kpt_label=True)
                output2 = output_to_keypoint(output_data2)

                img_ = image_[0].permute(1, 2, 0) * 255
                img_ = img_.cpu().numpy().astype(np.uint8)

                img_img = cv2.cvtColor(img_, cv2.COLOR_RGB2BGR)

                img = image[0].permute(1, 2, 0) * 255
                img = img.cpu().numpy().astype(np.uint8)

                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                img_2 = image_2[0].permute(1, 2, 0) * 255
                img_2 = img_2.cpu().numpy().astype(np.uint8)

                img_img2 = cv2.cvtColor(img_2, cv2.COLOR_RGB2BGR)
                height, width,_ = img_img2.shape
                img2 = image2[0].permute(1, 2, 0) * 255
                img2 = img2.cpu().numpy().astype(np.uint8)

                img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)
                #Plotting the Skeleton Keypoints on the Player 1
                for idx in range(output2.shape[0]):
                    plot_skeleton_kpts(img_img2, output2[idx, 7:].T, 3)

                gn = torch.tensor(img.shape)[[1, 0, 1, 0]]  # normalization gain whwh   
                for i, pose in enumerate(output_data):  # detections per image
                
                    if len(output_data):  #check if no pose
                        for c in pose[:, 5].unique(): # Print results
                            n = (pose[:, 5] == c).sum()  # detections per class
                            print("No of Objects in Current Frame : {}".format(n))
                        
                        for det_index, (*xyxy, conf, cls) in enumerate(reversed(pose[:,:6])): #loop over poses for drawing on frame
                            c = int(cls)  # integer class
                            kpts = pose[det_index, 6:]
                            label = names[c]
                            #This function will create a bounding box and assign a label to the player 2
                            plot_one_box_kpt(xyxy, img_img2, label=label, color=colors(c, True), 
                                        line_thickness=opt.line_thickness, kpts=kpts, steps=3, 
                                        orig_shape=img.shape[:2])                            
                            # == 6.0 === preprocess model input data and pose prediction =======
                            #1.So now, each video frame was fed into the YOLOv7 Pose Estimation Model 
                            #and predicted Key points Landmarks (X-coordinate, Y-coordinate and confidence) were extracted
                            # 2. and stacked together as a sequence of 30 frames.
                            #Here we will plot the skeleton keypoints on the Player 2

                            if j <= seq:
                                for idx in range(output.shape[0]):
                                    kpts = output[idx, 7:].T
                                    plot_skeleton_kpts(img_img2, kpts, 3)
                                    sequence.append(kpts.tolist())

                            # So if the length of the sequence is 30, then we will do the shot prediction,
                            # that whether it is a Smash Shot or it is a Dropped
                            if len(sequence) == 30:
                                # Doing the Shot Prediction over here
                                result = tf_model.predict(np.expand_dims(sequence, axis=0))
                                pose_name = actions[np.argmax(result)]
                                #So we will save all the sequence values which we have got 
                                keypoints.append(sequence)
                                #We are also saving all the Shot Names in a list as well
                                posename_list.append(pose_name)
                                print(sequence)
                                print(keypoints)
                                print(pose_name)
                                print(posename_list)

                            #And when the value of j becomes equal to the value of the sequence 
                            #which we have the defined as 30, then remove all the previous values of sequence
                            #and set it as an empty list and start processing on the next frames
                            
                            if j == seq:
                                sequence = []
                                j = 0
                            j += 1
                # =============================================================
                xstart = (fw//2)
                ystart = (fh-100)
                yend = (fh-50)
                # So After we have the shot name so here we are just setting the UI where we want to display the Shot names 
                # = 7.0 == Draw prediction ==================================
                if pose_name == "Smash-Shot":
                    cv2.line(img_img2, ((width - (width-50)),25), ((width - (width-200)),25), [85,45,255], 40)
                    cv2.putText(img_img2, "Shot Type", ((width - (width-50)),35), 0, 1, [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)
                    cv2.line(img_img2, ((width - (width-50)),75), ((width - (width-500)),75), [85,45,255], 40)
                    cv2.putText(img_img2, pose_name, ((width - (width-50)),85), 0, 1, [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)

                elif pose_name == "Drop-Shot":
                    cv2.line(img_img2, ((width - (width-50)),25), ((width - (width-200)),25), [85,45,255], 40)
                    cv2.putText(img_img2, "Shot Type", ((width - (width-50)),35), 0, 1, [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)
                    cv2.line(img_img2, ((width - (width-50)),75), ((width - (width-500)),75), [85,45,255], 40)
                    cv2.putText(img_img2, pose_name, ((width - (width-50)),85), 0, 1, [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)


                if webcam:
                    cv2.imshow("Detection", img_img2)
                    key = cv2.waitKey(1)
                    if key == ord('c'):
                        break
                else:
                    img_ = img.copy()
                    img_ = cv2.resize(
                        img_, (960, 540), interpolation=cv2.INTER_LINEAR)
                    cv2.imshow("Detection", img_img2)
                    cv2.waitKey(1)

                end_time = time.time()
                fps = 1 / (end_time - start_time)
                total_fps += fps
                frame_count += 1
                out.write(img_img2)
            else:
                break

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
