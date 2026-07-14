const { chromium } = require('playwright');

const TARGET_URL = 'http://k8s-mistudio.hitsai.local/';
const OUT = '/tmp/caps';
const fs = require('fs');
fs.mkdirSync(OUT, { recursive: true });

// sidebar label -> output filename
const PANELS = [
  ['Models', 'models'],
  ['Datasets', 'datasets'],
  ['Training', 'training'],
  ['Extractions', 'extractions'],
  ['Labeling', 'labeling'],
  ['Feature Groups', 'feature-groups'],
  ['SAEs', 'saes'],
  ['Steering', 'steering'],
  ['Templates', 'templates'],
  ['Monitor', 'monitor'],
  ['Settings', 'settings'],
];

(async () => {
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage();
  await page.setViewportSize({ width: 1600, height: 1000 });

  await page.goto(TARGET_URL, { waitUntil: 'networkidle', timeout: 30000 });
  console.log('Loaded:', await page.title());

  for (const [label, file] of PANELS) {
    try {
      const item = page.getByText(label, { exact: true }).first();
      await item.click({ timeout: 10000 });
      await page.waitForTimeout(3500); // let panel data load
      await page.screenshot({ path: `${OUT}/${file}.png`, fullPage: false });
      console.log(`OK  ${label} -> ${file}.png`);
    } catch (e) {
      console.error(`FAIL ${label}: ${e.message.split('\n')[0]}`);
    }
  }

  await browser.close();
  console.log('DONE');
})();
