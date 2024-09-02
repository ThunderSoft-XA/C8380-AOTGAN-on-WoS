from qai_appbuilder import (QNNContext, QNNContextProc, QNNShareMemory, Runtime, LogLevel, ProfilingLevel, PerfProfile, QNNConfig, timer)
from torchvision.transforms import ToTensor
import cv2, torch, numpy as np
import os, sys, time


ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
QNN_SDK_PATH = "C:\\Qualcomm\\AIStack\\QAIRT\\2.23.0.240531\\lib\\arm64x-windows-msvc"
MODEL_FILE = ROOT_DIR + "\\models\\aotgan.bin"
IMAGE_FILE = ROOT_DIR + "\\images\\face\\image\\imgHQ02076.png"
QNN_CONTEXT_PROC_USED = False
INFERENCE_TIME = 0.0


class Painter:
    def __init__(self, windowname, images, colors, thick, type):
        self.windowname = windowname
        self.images = images
        self.colors = colors
        self.thick = thick
        self.type = type
        self.prev_point = None
        self.dirty = False
        self.show()
        cv2.setMouseCallback(self.windowname, self.on_painted)

    def large_thick(self,):  
        self.thick = min(48, self.thick + 1)
    
    def small_thick(self,): 
        self.thick = max(3, self.thick - 1)

    def change_type(self,):
        if self.type == 'bbox':
            self.type = 'freeform'
        else:
            self.type = 'bbox'
        
    def show(self):
        cv2.imshow(self.windowname, self.images[0])

    def on_painted(self, event, x, y, flags, param):
        if self.type == 'bbox':
            self.on_bbox(event, x, y, flags, param)
        else:
            self.on_mouse(event, x, y, flags, param)

    def on_mouse(self, event, x, y, flags, param):
        global INFERENCE_TIME
        point = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            self.prev_point = point
        elif event == cv2.EVENT_LBUTTONUP:
            self.prev_point = None
        elif event == cv2.EVENT_MOUSEMOVE:
            cv2.rectangle(self.images[0], (0, 0), (270, 30), (0, 0, 0), -1)
            cv2.putText(self.images[0], f"({x}, {y}), {INFERENCE_TIME:.2f}ms", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            self.show()

        if self.prev_point and flags & cv2.EVENT_FLAG_LBUTTON:
            for image, color in zip(self.images, self.colors()):
                cv2.line(image, self.prev_point, point, color, self.thick)
            self.dirty = True
            self.prev_point = point
            self.show()

    def on_bbox(self, event, x, y, flags, param):
        global INFERENCE_TIME
        point = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            self.prev_point = point 
        elif event == cv2.EVENT_LBUTTONUP:
            for image, color in zip(self.images, self.colors()):
                cv2.rectangle(image, self.prev_point, point, color, -1)
            self.dirty = True
            self.prev_point = None
            self.show()
        elif event == cv2.EVENT_MOUSEMOVE:
            cv2.rectangle(self.images[0], (0, 0), (270, 30), (0, 0, 0), -1)
            cv2.putText(self.images[0], f"({x}, {y}), {INFERENCE_TIME:.2f}ms", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            self.show()


class AOTGAN1(QNNContext):
    def Inference(self, image, mask, perf_profile = PerfProfile.DEFAULT):
        input_data1 = image.permute(0, 2, 3, 1)
        input_data2 = mask.permute(0, 2, 3, 1)
        # input_data1 = np.transpose(image, (0, 2, 3, 1))
        # input_data2 = np.transpose(mask,  (0, 2, 3, 1))
        input_datas = [input_data1, input_data2]
        # print("input_data1.shape =", input_data1.shape)
        # print("input_data2.shape =", input_data2.shape)
        output_data = super().Inference(input_datas, perf_profile)[0]
        output_data = torch.from_numpy(output_data)
        output_data = output_data.reshape(1, 512, 512, 3)
        # print("output_data.shape =", output_data.shape)
        output_data = output_data.permute(0, 3, 1, 2)
        # output_data = np.transpose(output_data, (0, 3, 1, 2))
        return output_data

class AOTGAN2(QNNContextProc):
    def Inference(self, shared_memory, image, mask, perf_profile = PerfProfile.DEFAULT):
        input_data1 = image.permute(0, 2, 3, 1)
        input_data2 = mask.permute(0, 2, 3, 1)
        # input_data1 = np.transpose(image, (0, 2, 3, 1))
        # input_data2 = np.transpose(mask,  (0, 2, 3, 1))
        input_datas = [input_data1, input_data2]
        # print("input_data1.shape =", input_data1.shape)
        # print("input_data2.shape =", input_data2.shape)
        output_data = super().Inference(shared_memory, input_datas, perf_profile)[0]
        output_data = torch.from_numpy(output_data)
        output_data = output_data.reshape(1, 512, 512, 3)
        # print("output_data.shape =", output_data.shape)
        output_data = output_data.permute(0, 3, 1, 2)
        # output_data = np.transpose(output_data, (0, 3, 1, 2))
        return output_data


def postprocess(image):
    image = torch.clamp(image, -1., 1.)
    image = (image + 1) / 2.0 * 255.0
    image = image.permute(1, 2, 0)
    image = image.cpu().numpy().astype(np.uint8)
    return image


def main(qnn_sdk_path, model_file):
    global INFERENCE_TIME
    QNNConfig.Config(qnn_sdk_path, Runtime.HTP, LogLevel.WARN, ProfilingLevel.OFF)
    if QNN_CONTEXT_PROC_USED:
        shared_memory = QNNShareMemory("aotgan", 1024 * 1024 * 16)
        aotgan = AOTGAN2("aotgan", "aotgan", model_file)
    else:
        aotgan = AOTGAN1("aotgan", model_file)

    image_out_path = os.path.dirname(IMAGE_FILE)
    image_filename = os.path.basename(IMAGE_FILE).split('.')[0]
    orig_image = cv2.resize(cv2.imread(IMAGE_FILE, cv2.IMREAD_COLOR), (512, 512))
    # print("orig_image.shape =", orig_image.shape)

    h, w, _ = orig_image.shape
    image_mask = np.zeros([h, w, 1], np.uint8)
    image_copy = orig_image.copy()
    # print("image_mask.shape =", image_mask.shape)

    # image_tensor = (ToTensor()(orig_image) * 2.0 - 1.0).unsqueeze(0)                         # 0 bad convert!!!
    # tmp = torch.from_numpy(orig_image.transpose((2, 0, 1))).contiguous()                     # 1 bad convert!!!
    # image_tensor = (tmp.to(dtype=torch.float32).div(255) * 2.0 - 1.0).unsqueeze(0)           # 1 bad convert!!!
    # tmp = torch.from_numpy(orig_image.transpose((2, 0, 1)))                                  # 2 good convert!!!
    # image_tensor = (tmp.to(dtype=torch.float32).div(255) * 2.0 - 1.0).unsqueeze(0)           # 2 good convert!!!
    # rgb_image = orig_image / 255.0 * 2.0 - 1.0                                               # 3 good convert!!!
    # image_tensor = np.transpose(torch.from_numpy(rgb_image), (2, 0, 1)).unsqueeze(0).float() # 3 good convert!!!
    rgb_image = orig_image / 255.0 * 2.0 - 1.0                                                 # 4 good convert!!!
    image_tensor = torch.from_numpy(rgb_image).permute(2, 0, 1).unsqueeze(0).float()           # 4 good convert!!!
    # print("rgb_image.shape =", rgb_image.shape)
    # print("image_tensor.shape =", image_tensor.shape)

    painter = Painter("input image", [image_copy, image_mask], 
        lambda: ((255, 255, 255), (255, 255, 255)), 15, 'bbox')
    if not QNN_CONTEXT_PROC_USED:
        PerfProfile.SetPerfProfileGlobal(PerfProfile.BURST)

    while True:
        ch = cv2.waitKey()
        if ch == 27:
            print("quit!")
            break

        # process the image
        elif ch == ord(' '):
            print('[**] inpainting ... ')
            with torch.no_grad():
                # mask_tensor = (ToTensor()(image_mask)).unsqueeze(0)                                              # 0 good convert
                # mask_tensor = np.transpose(torch.from_numpy(image_mask / 255.0), (2, 0, 1)).unsqueeze(0).float() # 1 good convert
                mask_tensor = torch.from_numpy(image_mask / 255.0).permute(2, 0, 1).unsqueeze(0).float()           # 2 good convert
                masked_tensor = (image_tensor * (1 - mask_tensor).float()) + mask_tensor
                # print("masked_tensor.shape =", masked_tensor.shape)
                # print("mask_tensor.shape =", mask_tensor.shape)

                start_time = time.perf_counter()
                if QNN_CONTEXT_PROC_USED:
                    pred_tensor = aotgan.Inference(shared_memory, masked_tensor, mask_tensor, PerfProfile.BURST)
                else:
                    # PerfProfile.SetPerfProfileGlobal(PerfProfile.BURST)
                    pred_tensor = aotgan.Inference(masked_tensor, mask_tensor, PerfProfile.DEFAULT)
                    # PerfProfile.RelPerfProfileGlobal()
                end_time = time.perf_counter()
                INFERENCE_TIME = (end_time - start_time) * 1000
                # print("pred_tensor.shape=", pred_tensor.shape)

                comp_tensor = (pred_tensor * mask_tensor + image_tensor * (1 - mask_tensor))
                pred_np = postprocess(pred_tensor[0])
                masked_np = postprocess(masked_tensor[0])
                comp_np = postprocess(comp_tensor[0])

                cv2.imshow('inpainted image', comp_np)
                print(f"inpainting time: {INFERENCE_TIME:.3f} ms")
                print('inpainting finish!')

        # reset the mask
        elif ch == ord('r'):
            # image_tensor = (ToTensor()(orig_image) * 2.0 - 1.0).unsqueeze(0)                         # 0 bad convert
            # rgb_image = orig_image/ 255.0 * 2.0 - 1.0                                                # 1 bad convert
            # image_tensor = np.transpose(torch.from_numpy(rgb_image), (2, 0, 1)).unsqueeze(0).float() # 1 bad convert
            rgb_image = orig_image / 255.0 * 2.0 - 1.0                                                 # 2 bad convert
            image_tensor = torch.from_numpy(rgb_image).permute(2, 0, 1).unsqueeze(0).float()           # 2 bad convert
            image_copy[:] = orig_image.copy()
            image_mask[:] = 0
            painter.show()
            print("[**] reset!")

        elif ch == ord('k'): 
            print('[**] apply existing processing to images, and keep editing!')
            img_tensor = comp_tensor
            image_copy[:] = comp_np.copy()
            image_mask[:] = 0
            painter.show()
            print("reset!")
        
        elif ch == ord('+'): 
            painter.large_thick()

        elif ch == ord('-'): 
            painter.small_thick()

        # change the painter type
        elif ch == ord('c'): 
            painter.change_type()
        
        # save results
        if ch == ord('s'):
            cv2.imwrite(os.path.join(image_out_path, f'{image_filename}_masked.png'), masked_np)
            cv2.imwrite(os.path.join(image_out_path, f'{image_filename}_pred.png'),   pred_np)
            cv2.imwrite(os.path.join(image_out_path, f'{image_filename}_comp.png'),   comp_np)
            cv2.imwrite(os.path.join(image_out_path, f'{image_filename}_mask.png'),   image_mask)
            print('[**] save successfully!')

    if not QNN_CONTEXT_PROC_USED:
        PerfProfile.RelPerfProfileGlobal()
    cv2.destroyAllWindows()

print("Hello, AOTGAN!")
main(QNN_SDK_PATH, MODEL_FILE)
print("Bye, AOTGAN!")
