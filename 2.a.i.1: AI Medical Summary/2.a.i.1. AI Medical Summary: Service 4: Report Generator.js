import express from 'express';
import fs from 'fs-extra';
import path from 'path';
import handlebars from 'handlebars';
import puppeteer from 'puppeteer';
import { authMiddleware } from './auth.js';
import admin from 'firebase-admin';
import { Storage } from '@google-cloud/storage';
import { logAiServiceError } from './firebaseUtils.js';

// Firebase + GCS Init
// import fs from 'fs';

// Read the service account file from the path
const serviceAccountPath = process.env.FIREBASE_SERVICE_ACCOUNT;
const serviceAccount = JSON.parse(fs.readFileSync(serviceAccountPath, 'utf8'));

// Firebase Admin SDK Init
if (!admin.apps.length) {
  admin.initializeApp({
    credential: admin.credential.cert(serviceAccount),
    databaseURL: process.env.FIREBASE_DB_URL,
    storageBucket: '<some_storage_bucket>'
  });
}

// Google Cloud Storage Init
const gcs = new Storage({ credentials: serviceAccount });


export const renderRoute = express.Router();

renderRoute.post('/', authMiddleware, async (req, res) => {
  try {
    const { policyKey, fileAccessKey, jobType, caseFileID, file, mockFileCreation = false } = req.body;
    // Validate required fields
    const missingFields = [];
    if (!policyKey) missingFields.push('policyKey');
    if (!fileAccessKey) missingFields.push('fileAccessKey');
    if (!jobType) missingFields.push('jobType');
    if (!caseFileID) missingFields.push('caseFileID');
    if (!file) missingFields.push('file');

    if (missingFields.length > 0) {
      return res.status(400).json({
        error: `Missing required fields: ${missingFields.join(', ')}`
      });
    }
    console.log(`Received request for policy: ${policyKey} with fileAccessKey: ${fileAccessKey}`);
    console.log(`Case file ID: ${caseFileID} with job type: ${jobType}`);

    // Step 1: Read Firebase RTDB entry to get upload URL
    const dbRef = admin.database().ref(`fileAccess/${fileAccessKey}`);
    const fileAccessData = (await dbRef.once('value')).val();
    if (!fileAccessData || !fileAccessData.downloadURL) {
      throw new Error('Download URL not found in Firebase RTDB');
    }

    const downloadURL = fileAccessData.downloadURL;
    const tokenMatch = downloadURL.match(/token=([a-zA-Z0-9-]+)/);
    const tokenGuid = tokenMatch ? tokenMatch[1] : null;
    if (!tokenGuid) {
      throw new Error('Token GUID not found in download URL');
    }

    // Extract storage info from download URL
    const encodedPathMatch = downloadURL.match(/\/o\/(.*?)\?/);
    if (!encodedPathMatch || !encodedPathMatch[1]) {
      throw new Error('Failed to extract encoded path from downloadURL');
    }
    const decodedPath = decodeURIComponent(encodedPathMatch[1]);
    const outputFileName = path.basename(decodedPath);
    const storagePath = `policies/${policyKey}/${caseFileID}/${outputFileName}`;
    const bucket = admin.storage().bucket();
    const fileRef = bucket.file(storagePath);

    // ✅ MOCK FILE CREATION MODE
    if (mockFileCreation === true) {
      console.log('🧪 Running in mockFileCreation mode');
      const dummyPdf = Buffer.from(
        `%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n2 0 obj\n<< /Type /Pages /Count 1 /Kids [3 0 R] >>\nendobj\n3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 144] /Contents 4 0 R >>\nendobj\n4 0 obj\n<< /Length 44 >>\nstream\nBT /F1 24 Tf 100 100 Td (Mock PDF File) Tj ET\nendstream\nendobj\nxref\n0 5\n0000000000 65535 f\n0000000010 00000 n\n0000000053 00000 n\n0000000102 00000 n\n0000000175 00000 n\ntrailer\n<< /Root 1 0 R /Size 5 >>\nstartxref\n267\n%%EOF`,
        'utf-8'
      );

      await fileRef.save(dummyPdf, {
        metadata: {
          contentType: 'application/pdf',
          metadata: {
            firebaseStorageDownloadTokens: tokenGuid
          }
        }
      });
      console.log(`✅ Mock PDF uploaded to Firebase Storage at: ${storagePath}`);
      return res.status(200).json({
        file: downloadURL,
        message: 'Mock PDF file created successfully.'
      });
    }

    // ✅ ACTUAL PDF GENERATION FLOW (async)
    res.status(200).json({ file: downloadURL, message: 'PDF generation initiated.' });

    // === ASYNC LOGIC START ===
    // === ASYNC LOGIC START ===
    (async () => {
      if (!file.startsWith('gs://')) throw new Error('Input "file" must be a gs:// path');
      const bucketName = file.split('/')[2];
      const objectPath = file.split('/').slice(3).join('/');
      const remoteFile = gcs.bucket(bucketName).file(objectPath);
      const [contents] = await remoteFile.download();
      const data = JSON.parse(contents.toString());

      const fontPath = process.env.FONT1_PATH;
      const base64Font = fs.readFileSync(fontPath).toString('base64');
      const base64Image = (imgPath) => fs.readFileSync(imgPath).toString('base64');

      let pageImages = {};
      let templateHtml = "";

      // ✅ Job-specific initialization
      if (jobType === "aiLEArbitrage") {
        // Only 1 page for aiLEArbitrage
        pageImages = {
          page1: `data:image/png;base64,${base64Image(process.env.ARB_PAGE1_IMAGE)}`,
          page2: `data:image/png;base64,${base64Image(process.env.ARB_PAGE2_IMAGE)}`
        };
        templateHtml = await fs.readFile(process.env.ARB_HBS_TEMPLATE, 'utf8');
      } 
      else if (jobType === "aiMedicalSummary") {
        // Multiple pages for aiMedicalSummary
        pageImages = {
          page1: `data:image/png;base64,${base64Image(process.env.PAGE1_IMAGE)}`,
          page2: `data:image/png;base64,${base64Image(process.env.PAGE2_IMAGE)}`,
          page3: `data:image/png;base64,${base64Image(process.env.PAGE3_IMAGE)}`,
          page4: `data:image/png;base64,${base64Image(process.env.PAGE4_IMAGE)}`,
          page5: `data:image/png;base64,${base64Image(process.env.PAGE5_IMAGE)}`,
          page12: `data:image/png;base64,${base64Image(process.env.PAGE12_IMAGE)}`
        };
        templateHtml = await fs.readFile(process.env.HBS_TEMPLATE, 'utf8');
      } 
      else {
        throw new Error(`Unsupported jobType: ${jobType}`);
      }

      const template = handlebars.compile(templateHtml);
      const html = template({ ...data, ...pageImages, customFontBase64: base64Font });

      const browser = await puppeteer.launch({ headless: true, args: ['--no-sandbox'] });
      const page = await browser.newPage();
      await page.setViewport({
        width: 893,
        height: 1259,
        deviceScaleFactor: 1,
      });

      await page.setContent(html, { waitUntil: 'networkidle0' });
      const pdfBuffer = await page.pdf({
        format: 'Letter',
        printBackground: true,
        margin: { top: '1in', bottom: '1in', left: '0.75in', right: '0.75in' }
      });
      await browser.close();

      await fileRef.save(pdfBuffer, {
        metadata: {
          contentType: 'application/pdf',
          metadata: {
            firebaseStorageDownloadTokens: tokenGuid
          }
        }
      });

      console.log(`✅ PDF uploaded to Firebase Storage at: ${storagePath}`);
    })().catch(async (err) => {
      console.error(`❌ Async PDF generation failed: ${err}`);
      await logAiServiceError({
        jobType: jobType,
        caseFileID: caseFileID,
        error: err,
        requestData: req.body
      });
    });
    // === ASYNC LOGIC END ===

  } catch (err) {
    console.error("Inside synchronous reportGenerator error handling.")
    console.error(err);
    const { jobType, caseFileID } = req.body;
    await logAiServiceError({
      jobType: jobType,
      caseFileID: caseFileID,
      error: err,
      requestData: req.body
    });
    res.status(500).json({ error: 'Failed to generate PDF' });
  }
});

