/* Run with Playwright available, static docs on :8001 and API on :8002.
   Uses a generated one-pixel PNG only; no photos, OpenAI or Box calls. */
const { chromium } = require('playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');

(async () => {
  const browser = await chromium.launch({headless: true, executablePath: process.env.CHROMIUM_PATH || '/usr/bin/chromium'});
  try {
    const page = await browser.newPage({viewport: {width: 1100, height: 1000}});
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    let generationCalls = 0;
    let timelineCalls = 0;
    let failPreview = false;
    const pixel = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII=';
    await page.route('http://127.0.0.1:8000/api/**', async route => {
      const request = route.request();
      if (request.url().endsWith('/api/timeline')) {
        timelineCalls++;
        assert(!request.postData().includes('image'));
        if (failPreview) return route.fulfill({status: 503, json: {detail: 'Test outage'}});
        const response = await page.request.post('http://127.0.0.1:8002/api/timeline', {data: request.postDataJSON()});
        return route.fulfill({response});
      }
      assert(request.url().endsWith('/api/generate'));
      generationCalls++;
      const profile = {age: 20, target_age: 65, sex: '', ethnicity: '', factors: {smoking: 0, uv_exposure: .4, alcohol: .1, stress: .6, diet: .3}};
      const preview = await page.request.post('http://127.0.0.1:8002/api/timeline', {data: {profile, stages: 1}});
      const data = await preview.json();
      return route.fulfill({json: {...data.snapshots[0], image: {mime_type: 'image/png', base64: pixel}, storage_status: 'disabled'}});
    });
    await page.goto('http://127.0.0.1:8001');
    // Existing pre-IRB consent behavior is unchanged; preview needs no photo.
    await page.locator('#preview-button').click();
    await page.waitForFunction(() => document.querySelector('#knowledge-status').textContent.includes('age 65.'));
    const oldFeatures = await page.locator('#feature-list').innerText();
    await page.locator('#timeline-age').fill('40');
    await page.waitForFunction(() => document.querySelector('#knowledge-status').textContent.includes('age 40.'));
    assert.notEqual(await page.locator('#feature-list').innerText(), oldFeatures);
    assert.equal(generationCalls, 0);
    assert(await page.locator('#aging-layers details').count() > 0);
    await page.locator('#aging-layers details').first().evaluate(el => {el.open = true;});
    assert((await page.locator('#aging-layers').innerText()).includes('not a medical prediction'));
    fs.mkdirSync('docs/screenshots', {recursive: true});
    await page.locator('#knowledge-preview').screenshot({path: 'docs/screenshots/knowledge-timeline.png'});
    // Mock one explicit generation, then show that slider leaves the image alone.
    await page.locator('#image').setInputFiles({name: 'synthetic-pixel.png', mimeType: 'image/png', buffer: Buffer.from(pixel, 'base64')});
    await page.locator('#generate-button').click();
    await page.waitForFunction(() => document.querySelector('#portrait-label').textContent.includes('age 65.'));
    const image = await page.locator('#aged-result').getAttribute('src');
    await page.locator('#timeline-age').fill('75');
    await page.waitForFunction(() => document.querySelector('#knowledge-status').textContent.includes('age 75.'));
    assert.equal(await page.locator('#aged-result').getAttribute('src'), image);
    assert((await page.locator('#portrait-label').innerText()).includes('age 65.'));
    assert(await page.locator('#feedback-button').isDisabled());
    assert.equal(generationCalls, 1);
    // Rapid age changes must settle on the last age.
    await page.locator('#timeline-age').fill('55');
    await page.locator('#timeline-age').fill('80');
    await page.waitForFunction(() => document.querySelector('#knowledge-status').textContent.includes('age 80.'));
    failPreview = true;
    await page.locator('#timeline-age').fill('81');
    await page.waitForFunction(() => document.querySelector('#knowledge-status').textContent.includes('Preview unavailable: Test outage'));
    assert.equal(generationCalls, 1);
    assert(timelineCalls >= 5);
    assert.deepEqual(errors, []);
    // Mobile layout must not overflow.
    await page.setViewportSize({width: 390, height: 844});
    assert(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth));
    console.log('Frontend passed: no-photo preview, real age scores/Layers, immutable portrait, storage feedback gate, latest age, error handling, mobile layout.');
  } finally { await browser.close(); }
})().catch(error => {console.error(error); process.exit(1);});
