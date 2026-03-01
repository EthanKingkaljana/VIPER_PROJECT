// File: src/lib/templates/enhancement.ts
import type { WorkflowTemplate } from '../workflowTemplates';
import type { Node } from 'reactflow';

/* =========================================================
   SHARED SAMPLE IMAGES
========================================================= */

const SAMPLE_IMG = '/static/samples/image29-261x300.jpg';
const LOWLIGHT_IMG = '/static/samples/lowlight.jpg';

/* =========================================================
   CLAHE TEMPLATE
========================================================= */

const CLAHE_INPUT_NODE: Node = {
  id: 'n1-enhance-clahe',
  type: 'image-input',
  position: { x: 250, y: 50 },
  data: {
    label: 'Input Image',
    status: 'success',
    description: "Sample image loaded successfully.",
    payload: {
      name: 'lele.jpg',
      url: SAMPLE_IMG,
      width: 512,
      height: 512
    }
  }
};

export const ENHANCEMENT_CLAHE_TEMPLATE: WorkflowTemplate = {
  name: 'CLAHE (Contrast Limited Adaptive Histogram Equalization)',

  descriptor: {
    en: 'Enhance image contrast using adaptive histogram equalization.',
    th: 'เพิ่มความคมชัดของภาพด้วยเทคนิคปรับฮิสโตแกรมแบบปรับตามพื้นที่'
  },

  description: 'Input Image → CLAHE Enhancement',

  longDescription: {
    en: `
This workflow applies CLAHE (Contrast Limited Adaptive Histogram Equalization) 
to improve local contrast in an image.

CLAHE enhances visibility in low-contrast regions while preventing over-amplification 
of noise by limiting contrast enhancement.

Best suited for:
• Medical imaging
• Low-light photographs
• X-ray or grayscale images
• Detail enhancement tasks

Adjustable parameters:
• clipLimit – Controls contrast amplification limit
• tileGridSize – Defines grid size for local histogram equalization
    `,
    th: `
เวิร์กโฟลว์นี้ใช้เทคนิค CLAHE เพื่อเพิ่มความคมชัดแบบเฉพาะพื้นที่

ช่วยเพิ่มรายละเอียดในบริเวณที่มีความเปรียบต่างต่ำ
และป้องกันการขยายสัญญาณรบกวนมากเกินไป

เหมาะสำหรับ:
• ภาพทางการแพทย์
• ภาพแสงน้อย
• ภาพเอกซเรย์
• งานเพิ่มรายละเอียดภาพ

พารามิเตอร์:
• clipLimit – ควบคุมระดับการเพิ่มความคมชัด
• tileGridSize – ขนาดพื้นที่ย่อยสำหรับคำนวณฮิสโตแกรม
    `
  },

  color: 'indigo',

  nodes: [
    CLAHE_INPUT_NODE,
    {
      id: 'n2-clahe',
      type: 'clahe',
      position: { x: 650, y: 200 },
      data: {
        label: 'CLAHE Enhancement',
        status: 'idle',
        description: "Ready to enhance image contrast using CLAHE.",
        payload: {
          params: {
            clipLimit: 2.0,
            tileGridSizeX: 8,
            tileGridSizeY: 8
          }
        }
      }
    } as Node,
  ],

  edges: [
    {
      id: 'e1-clahe',
      source: 'n1-enhance-clahe',
      target: 'n2-clahe',
      type: 'smoothstep',
      style: { stroke: "#64748b", strokeWidth: 2 }
    },
  ]
};

/* =========================================================
   ZERO-DCE TEMPLATE
========================================================= */

const ZERODCE_INPUT_NODE: Node = {
  id: 'n1-enhance-zerodce',
  type: 'image-input',
  position: { x: 250, y: 50 },
  data: {
    label: 'Input Image',
    status: 'success',
    description: "Low-light sample image loaded successfully.",
    payload: {
      name: 'lowlight.jpg',
      url: LOWLIGHT_IMG,
      width: 512,
      height: 512
    }
  }
};

export const ENHANCEMENT_ZERODCE_TEMPLATE: WorkflowTemplate = {
  name: 'Zero-DCE (Low-Light Enhancement)',

  descriptor: {
    en: 'Enhance low-light images using Zero-Reference Deep Curve Estimation.',
    th: 'ปรับปรุงภาพแสงน้อยด้วยเทคนิค Zero-Reference Deep Curve Estimation'
  },

  description: 'Input Image → Zero-DCE Enhancement',

  longDescription: {
    en: `
This workflow applies Zero-DCE (Zero-Reference Deep Curve Estimation),
a deep learning-based method for low-light image enhancement.

Unlike traditional enhancement techniques, Zero-DCE does not require paired training data.
It estimates pixel-wise light enhancement curves directly from a single image.

Zero-DCE improves:
• Brightness in dark areas
• Global and local contrast
• Color consistency
• Illumination balance

Ideal for:
• Night photography
• Underexposed images
• Surveillance footage
• Mobile low-light captures

Adjustable parameters:
• iterations – Number of curve refinement iterations (1–16)
    `,
    th: `
เวิร์กโฟลว์นี้ใช้โมเดล Zero-DCE สำหรับปรับปรุงภาพในสภาพแสงน้อย

ไม่ต้องใช้ภาพคู่สำหรับการฝึก
และคำนวณเส้นโค้งปรับแสงในระดับพิกเซลโดยตรง

ช่วย:
• เพิ่มความสว่างในบริเวณมืด
• เพิ่มความคมชัด
• ปรับสีให้สมดุล
• ทำให้แสงดูเป็นธรรมชาติ

เหมาะสำหรับ:
• ภาพกลางคืน
• ภาพแสงน้อย
• ภาพจากกล้องวงจรปิด
• ภาพมือถือในที่มืด

พารามิเตอร์:
• iterations – จำนวนรอบการปรับแสง (1–16)
    `
  },

  color: 'indigo',

  nodes: [
    ZERODCE_INPUT_NODE,
    {
      id: 'n2-zerodce',
      type: 'zero',
      position: { x: 650, y: 200 },
      data: {
        label: 'Zero-DCE Enhancement',
        status: 'idle',
        description: "Ready to enhance low-light image.",
        payload: {
          params: {
            iterations: 8
          }
        }
      }
    } as Node,
  ],

  edges: [
    {
      id: 'e1-zerodce',
      source: 'n1-enhance-zerodce',
      target: 'n2-zerodce',
      type: 'smoothstep',
      style: { stroke: "#64748b", strokeWidth: 2 }
    },
  ]
};
