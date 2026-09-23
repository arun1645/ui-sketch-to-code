const fs = require('fs');
const path = require('path');
const { JSDOM } = require('jsdom');
const createDOMPurify = require('dompurify');
const { HtmlValidate } = require('html-validate');

console.log('=== Node.js Dependencies Verification ===');

// 1. Verify DOMPurify + JSDOM
try {
  const window = new JSDOM('').window;
  const DOMPurify = createDOMPurify(window);
  const dirty = '<script>alert("xss")</script><div class="card"><h1>Header</h1></div>';
  const clean = DOMPurify.sanitize(dirty);
  console.log('[OK] DOMPurify sanitization successful.');
  console.log('     Input:', dirty);
  console.log('     Output:', clean);
} catch (err) {
  console.error('[ERROR] DOMPurify failed:', err.message);
  process.exit(1);
}

// 2. Verify html-validate
try {
  const htmlvalidate = new HtmlValidate();
  const testHtml = '<!DOCTYPE html>\n<html lang="en"><head><title>Test</title></head><body><div>Valid HTML</div></body></html>';
  const report = htmlvalidate.validateStringSync(testHtml);
  console.log('[OK] html-validate check completed. Valid structure:', report.valid);
} catch (err) {
  console.error('[ERROR] html-validate failed:', err.message);
  process.exit(1);
}

console.log('=== All Node.js dependencies verified successfully! ===');
