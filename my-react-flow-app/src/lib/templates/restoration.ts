// File: src/lib/templates/restoration.ts
import type { WorkflowTemplate } from '../workflowTemplates';
import type { Node } from 'reactflow';

const SAMPLE_IMG = '/static/samples/64x64.png';
const SAMPLE_IMG_DNCNN = '/static/samples/denoise_sample.jpg';

const INPUT_NODE: Node = {
  id: 'n1-restore',
  type: 'image-input',
  position: { x: 250, y: 50 },
  data: {
    label: 'Input Image',
    status: 'success',
    description: "Low-resolution sample image loaded successfully.",
    payload: {
      name: 'sample_64x64.png',
      url: SAMPLE_IMG,
      width: 64,
      height: 64
    }
  }
};

export const RESTORATION_REALESRGAN_TEMPLATE: WorkflowTemplate = {
  name: 'Real-ESRGAN (Super Resolution & Restoration)',

  descriptor: {
    en: 'Enhance image resolution and restore fine details using AI super-resolution.',
    th: 'เพิ่มความละเอียดและฟื้นฟูรายละเอียดของภาพด้วยเทคโนโลยี AI Super-Resolution'
  },

  description: 'Input Image → Real-ESRGAN Super Resolution',

  longDescription: {
    en: `
This workflow applies Real-ESRGAN (Enhanced Super-Resolution Generative Adversarial Network)
to upscale and restore low-resolution images using deep learning.

Real-ESRGAN enhances image quality by:
• Increasing resolution (2x, 4x, 8x)
• Recovering fine textures and edges
• Reducing compression artifacts
• Suppressing noise

Ideal for:
• Old photographs
• Low-resolution thumbnails
• AI-generated images
• Surveillance or compressed images

Adjustable parameters:
• scale – Upscaling factor (2x, 4x, 8x)
• denoise – Noise reduction strength (0.0–1.0)
• model_name – Pretrained model selection
    `,
    th: `
เวิร์กโฟลว์นี้ใช้เทคนิค Real-ESRGAN (Enhanced Super-Resolution Generative Adversarial Network)
เพื่อขยายความละเอียดและฟื้นฟูภาพความละเอียดต่ำด้วย Deep Learning

Real-ESRGAN สามารถ:
• ขยายความละเอียดของภาพ (2x, 4x, 8x)
• คืนรายละเอียดพื้นผิวและขอบภาพ
• ลดรอยบีบอัด (Compression Artifacts)
• ลดสัญญาณรบกวนในภาพ

เหมาะสำหรับ:
• ภาพเก่าหรือภาพความละเอียดต่ำ
• ภาพขนาดเล็ก (Thumbnail)
• ภาพที่ถูกบีบอัด
• ภาพจากกล้องวงจรปิด

พารามิเตอร์ที่ปรับได้:
• scale – ระดับการขยายภาพ (2x, 4x, 8x)
• denoise – ความแรงของการลด Noise (0.0–1.0)
• model_name – เลือกโมเดลที่ใช้สำหรับการประมวลผล
    `
  },

  color: 'red',

  nodes: [
    INPUT_NODE,
    {
      id: 'n2-realesrgan',
      type: 'realesrgan',
      position: { x: 650, y: 100 },
      data: {
        label: 'Real-ESRGAN Upscaler',
        status: 'idle',
        description: "Ready to upscale and restore image quality.",
        payload: {
          params: {
            scale: 4,
            denoise: 0.4,
            model_name: 'RealESRGAN_x4plus'
          }
        }
      }
    } as Node,
  ],

  edges: [
    {
      id: 'e1',
      source: 'n1-restore',
      target: 'n2-realesrgan',
      type: 'smoothstep',
      style: { stroke: "#64748b", strokeWidth: 2 }
    },
  ]
};

/* =========================
   DnCNN Template
========================= */

const INPUT_NODE_DNCNN: Node = {
  id: 'n1-dncnn',
  type: 'image-input',
  position: { x: 250, y: 50 },
  data: {
    label: 'Input Image',
    status: 'success',
    description: "Noisy image loaded successfully.",
    payload: {
      name: 'denoise_sample.jpg',
      url: SAMPLE_IMG_DNCNN,
      width: 512,
      height: 512,
    }
  }
};

export const RESTORATION_DNCNN_TEMPLATE: WorkflowTemplate = {
  name: 'DnCNN (Image Denoising)',

  descriptor: {
    en: 'Remove image noise using deep convolutional neural network (DnCNN).',
    th: 'ลดสัญญาณรบกวนของภาพด้วยโมเดล Deep Convolutional Neural Network (DnCNN)',
  },

  description: 'Input Image → DnCNN Denoising',

  longDescription: {
    en: `
This workflow applies DnCNN (Denoising Convolutional Neural Network)
to remove noise from images using deep learning.

DnCNN works by learning the residual noise pattern instead of directly
predicting the clean image. This residual learning strategy makes
training faster and more stable.

DnCNN is effective for:
• Gaussian noise removal
• Real-world noise reduction
• Preprocessing before segmentation or detection
• Improving visual clarity

Adjustable parameters:
• noise_level – Expected noise intensity
• model_name – Pretrained denoising model
    `,
    th: `
เวิร์กโฟลว์นี้ใช้โมเดล DnCNN (Denoising Convolutional Neural Network)
สำหรับลดสัญญาณรบกวนในภาพด้วย Deep Learning

DnCNN ทำงานโดยเรียนรู้ "Residual Noise"
แทนการทำนายภาพที่สะอาดโดยตรง
ซึ่งช่วยให้โมเดลเรียนรู้ได้เร็วและมีเสถียรภาพสูง

DnCNN เหมาะสำหรับ:
• การลบ Gaussian noise
• การลด noise จากภาพจริง
• ใช้เป็นขั้นตอนเตรียมภาพก่อนทำ Segmentation หรือ Detection
• เพิ่มความคมชัดของภาพ

พารามิเตอร์ที่ปรับได้:
• noise_level – ระดับความแรงของ noise
• model_name – โมเดลที่ใช้ในการลด noise
    `,
  },

  color: 'red',

  nodes: [
    INPUT_NODE_DNCNN,
    {
      id: 'n2-dncnn',
      type: 'dncnn',
      position: { x: 650, y: 300 },
      data: {
        label: 'DnCNN Denoiser',
        status: 'idle',
        description: 'Ready to remove image noise.',
        payload: {
          params: {
            noise_level: 25,
            model_name: 'DnCNN-S-25',
          },
        },
      },
    } as Node,
  ],

  edges: [
    {
      id: 'e1-dncnn',
      source: 'n1-dncnn',
      target: 'n2-dncnn',
      type: 'smoothstep',
      style: { stroke: '#64748b', strokeWidth: 2 },
    },
  ],
};