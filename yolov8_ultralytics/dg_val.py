"""
DG version of validation PyTorch yolov8-nano 
"""
import argparse
from ultralytics import YOLO

def parser_arguments():
    """
    Parser section
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, help='initial weights path')
    parser.add_argument('--data', type=str, default='coco128.yaml', help='dataset.yaml path')
    parser.add_argument('--annotations', type=str, default=None, help='ground truth annotation json file path')
    #parser.add_argument('--device', type=str, default='cpu', help='device to use')
    parser.add_argument('--device', type=str, default='0', help='device to use')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train, val image size (pixels)')
    parser.add_argument('--no-separate-outputs', action='store_true', help='exported file without separate outputs')

    return parser.parse_args()

if __name__ == '__main__':

    args = parser_arguments()
    # Create a dictionary of kwargs
    kwargs = vars(args)

    print(kwargs)

    model = YOLO(args.weights)

    SEPERATE_OUTPUTS = False if (args.weights.endswith('.pt') or args.no_separate_outputs) else True
    SAVE_JSON = True if args.annotations else False
    success = model.val(cache=False, int8=True, batch=16, data=args.data, imgsz=args.imgsz, rect=False, device=args.device, separate_outputs=SEPERATE_OUTPUTS, save_json=SAVE_JSON, anno_json=args.annotations)
