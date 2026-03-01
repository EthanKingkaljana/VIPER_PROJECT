// File: src/lib/templates/segmentation.ts
import type { WorkflowTemplate } from '../workflowTemplates';
import type { Node } from 'reactflow';

/* =========================
   Sample Images
========================= */
const SAMPLE_IMG_DEEPLAB = '/static/samples/images (5).jpg';
const SAMPLE_IMG_MASKRCNN = '/static/samples/maskrcnn_sample.jpg';

/* =========================
   Input Nodes (SEPARATE)
========================= */

// DeepLab input node
const INPUT_NODE_DEEPLAB: Node = {
  id: 'n1-deeplab',
  type: 'image-input',
  position: { x: 250, y: 50 },
  data: {
    label: 'Input Image',
    status: 'success',
    description: 'DeepLab sample image loaded successfully.',
    payload: {
      name: 'deeplab_sample.jpg',
      url: SAMPLE_IMG_DEEPLAB,
      width: 800,
      height: 600,
    },
  },
};

// Mask R-CNN input node
const INPUT_NODE_MASKRCNN: Node = {
  id: 'n1-mask',
  type: 'image-input',
  position: { x: 250, y: 50 },
  data: {
    label: 'Input Image',
    status: 'success',
    description: 'Mask R-CNN sample image loaded successfully.',
    payload: {
      name: 'maskrcnn_sample.jpg',
      url: SAMPLE_IMG_MASKRCNN,
      width: 800,
      height: 600,
    },
  },
};

/* =========================
   DeepLab Template
========================= */

export const SEGMENTATION_DEEPLAB_TEMPLATE: WorkflowTemplate = {
  name: 'DeepLab V3+ (Semantic Segmentation)',

  descriptor: {
    en: 'Perform pixel-level semantic segmentation using DeepLab V3+.',
    th: 'แยกส่วนภาพแบบกำหนดประเภทในระดับพิกเซลด้วยโมเดล DeepLab V3+',
  },

  description: 'Input Image → DeepLab V3+ Segmentation',

  longDescription: {
    en: `
DeepLabV3+ is a powerful algorithm for performing semantic segmentation in an image.
It classifies every pixel into a predefined category, allowing the system to understand
what each region of the image represents (such as road, sky, person, or car).

It works by using atrous (dilated) convolutions to capture multi-scale contextual
information without reducing spatial resolution. DeepLabV3+ also includes an
encoder–decoder architecture that refines object boundaries for more precise results.

Unlike object detection models, DeepLabV3+ does not separate individual objects —
it labels all pixels belonging to the same class as one region.

Similar algorithms in this category include FCN, U-Net, and PSPNet.
    `,
    th: `
DeepLabV3+ เป็นอัลกอริทึมสำหรับงาน Semantic Segmentation
ซึ่งทำหน้าที่จำแนกประเภทของทุกพิกเซลในภาพ ทำให้ระบบเข้าใจได้ว่า
แต่ละพื้นที่ของภาพคืออะไร เช่น ถนน ท้องฟ้า คน หรือรถยนต์

โมเดลใช้ Atrous (Dilated) Convolution เพื่อเก็บข้อมูลหลายระดับความละเอียด
โดยไม่ลดความคมชัดของภาพ และใช้โครงสร้างแบบ Encoder–Decoder
เพื่อปรับปรุงขอบวัตถุให้แม่นยำมากขึ้น

ต่างจากโมเดลตรวจจับวัตถุ DeepLabV3+ จะไม่แยกวัตถุเป็นรายชิ้น
แต่จะรวมพิกเซลที่อยู่ในคลาสเดียวกันเป็นพื้นที่เดียวกัน

โมเดลที่คล้ายกัน ได้แก่ FCN, U-Net และ PSPNet
    `,
  },

  color: 'yellow',

  nodes: [
    INPUT_NODE_DEEPLAB,
    {
      id: 'n2-deeplab',
      type: 'deeplab',
      position: { x: 650, y: 300 },
      data: {
        label: 'DeepLab V3+',
        status: 'idle',
        description: 'Ready to perform semantic segmentation.',
        payload: {
          params: {
            backbone: 'resnet50',
            dataset: 'coco',
            output_stride: 16,
          },
        },
      },
    } as Node,
  ],

  edges: [
    {
      id: 'e1-deeplab',
      source: 'n1-deeplab',
      target: 'n2-deeplab',
      type: 'smoothstep',
      style: { stroke: '#64748b', strokeWidth: 2 },
    },
  ],
};

/* =========================
   Mask R-CNN Template
========================= */

export const SEGMENTATION_MASKRCNN_TEMPLATE: WorkflowTemplate = {
  name: 'Mask R-CNN (Instance Segmentation)',

  descriptor: {
    en: 'Perform instance-level segmentation using Mask R-CNN.',
    th: 'แยกวัตถุแบบรายชิ้น (Instance Segmentation) ด้วยโมเดล Mask R-CNN',
  },

  description: 'Input Image → Mask R-CNN Instance Segmentation',

  longDescription: {
    en: `
Mask R-CNN is a powerful algorithm for instance segmentation in an image.
It detects individual objects and generates a precise segmentation mask
for each detected object separately.

It extends Faster R-CNN by adding a parallel branch that predicts
a binary mask for every Region of Interest (RoI). The model performs:

• Object detection (bounding boxes)
• Object classification
• Pixel-level mask prediction per instance

Unlike semantic segmentation models, Mask R-CNN separates different
objects even if they belong to the same class (e.g., multiple people).

Similar algorithms in this category include Faster R-CNN, YOLO (for detection),
and SOLO (Segmenting Objects by Locations).
    `,
    th: `
Mask R-CNN เป็นอัลกอริทึมสำหรับงาน Instance Segmentation
ซึ่งสามารถตรวจจับวัตถุแต่ละชิ้นในภาพและสร้างหน้ากากแยกให้แต่ละวัตถุได้อย่างแม่นยำ

โมเดลพัฒนาต่อจาก Faster R-CNN โดยเพิ่มสาขา (branch) สำหรับทำนาย
หน้ากากแบบพิกเซลต่อวัตถุแต่ละชิ้น พร้อมกับ:

• ตรวจจับกรอบวัตถุ (Bounding Box)
• จำแนกประเภทวัตถุ
• สร้างหน้ากากแยกรายวัตถุ

แตกต่างจาก Semantic Segmentation ตรงที่ Mask R-CNN
จะแยกวัตถุแต่ละชิ้นออกจากกัน แม้จะอยู่ในคลาสเดียวกัน
เช่น คนหลายคนในภาพเดียวกัน

โมเดลที่เกี่ยวข้อง ได้แก่ Faster R-CNN, YOLO และ SOLO
    `,
  },

  color: 'yellow',

  nodes: [
    INPUT_NODE_MASKRCNN,
    {
      id: 'n2-maskrcnn',
      type: 'maskrcnn',
      position: { x: 650, y: 300 },
      data: {
        label: 'Mask R-CNN',
        status: 'idle',
        description: 'Ready to perform instance segmentation.',
        payload: {
          params: {
            backbone: 'resnet50',
            dataset: 'coco',
            score_threshold: 0.5,
          },
        },
      },
    } as Node,
  ],

  edges: [
    {
      id: 'e1-mask',
      source: 'n1-mask',
      target: 'n2-maskrcnn',
      type: 'smoothstep',
      style: { stroke: '#64748b', strokeWidth: 2 },
    },
  ],
};