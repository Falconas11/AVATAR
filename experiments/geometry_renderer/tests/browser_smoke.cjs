/* Optional real-browser validation. Requires Playwright and a running demo :8501.
   This explicitly screenshots ONLY the public-domain NASA fixture, never uploads. */
const {chromium} = require('playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

(async () => {
  const browser = await chromium.launch({headless: true, executablePath: process.env.CHROMIUM_PATH || '/usr/bin/chromium'});
  try {
    const page = await browser.newPage({viewport: {width: 1500, height: 1500}});
    const externalRequests = [], errors = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.route('**/*', route => {
      const hostname = new URL(route.request().url()).hostname;
      if (hostname && !['127.0.0.1', 'localhost'].includes(hostname)) {
        externalRequests.push(hostname);
        return route.abort();
      }
      return route.continue();
    });
    const start = Date.now();
    await page.goto('http://127.0.0.1:8501');
    const sample = page.getByText('Use public-domain NASA sample', {exact: true});
    await sample.click();
    await page.getByText(/detection passes for this image: 1/).waitFor();
    const loadMs = Date.now()-start;
    const metrics = page.locator('[data-testid=stMetricValue]');
    assert(parseFloat(await metrics.first().innerText()) > 1);
    await page.locator('[data-testid=stImage] img').first().evaluate(img => img.decode());
    assert(await page.locator('[data-testid=stImage] img').first().evaluate(img => img.naturalWidth < 256));
    const slider = page.getByRole('slider', {name: 'Illustrative target age', exact: true});
    await slider.focus();
    const warpStart = Date.now();
    await page.keyboard.press('ArrowRight');
    await page.getByRole('heading', {name: 'Illustrative geometry — selected age 66', exact: true}).waitFor();
    const interactiveMs = Date.now()-warpStart;
    assert.equal(await page.getByText(/detection passes for this image: 1/).count(), 1);
    const output = path.resolve(__dirname, '../screenshots');
    fs.mkdirSync(output, {recursive: true});
    await page.locator('[data-testid=stImage] img').last().evaluate(img => img.decode());
    await page.screenshot({path: path.join(output, 'demo.png'), fullPage: true});
    const previousImage = await page.locator('[data-testid=stImage] img').last().getAttribute('src');
    await page.getByText('Show mesh and landmarks', {exact: true}).click();
    // Wait for new image media, not just a checkbox or a transient run status.
    await page.waitForFunction(previous => {
      const imgs = document.querySelectorAll('[data-testid=stImage] img');
      return imgs.length === 2 && imgs[1].getAttribute('src') !== previous && [...imgs].every(img => img.complete && img.naturalWidth > 0);
    }, previousImage);
    await page.screenshot({path: path.join(output, 'mesh.png'), fullPage: true});
    // Inspection modes reuse the detected mesh and do not add deformation.
    await page.getByRole('combobox').click();
    await page.getByRole('option', {name: 'Before / after wipe', exact: true}).click();
    await page.getByRole('heading', {name: 'Aligned before / after comparison'}).waitFor();
    assert.equal(await page.getByText(/detection passes for this image: 1/).count(), 1);
    await page.getByRole('combobox').click();
    await page.getByRole('option', {name: 'Pixel difference ×4', exact: true}).click();
    await page.getByRole('heading', {name: 'Pixel difference ×4 — diagnostic view', exact: true}).waitFor();
    await page.getByText('Focus on face', {exact: true}).click();
    await page.waitForFunction(() => {const img=document.querySelector('[data-testid=stImage] img');return img && img.complete && img.naturalWidth===512;});
    await page.getByRole('combobox').click();
    await page.getByRole('option', {name: 'Side by side', exact: true}).click();
    await page.getByRole('button', {name: 'Reset deformation', exact: true}).click();
    await page.getByRole('heading', {name: 'Illustrative geometry — selected age 30', exact: true}).waitFor();
    await page.getByRole('button', {name: 'Clear image from this session', exact: true}).click();
    await page.getByText(/Select a local portrait with exactly one/).waitFor();
    // Corrupt file handled locally with a safe error rather than a traceback.
    await page.locator('input[type=file]').setInputFiles({name:'invalid.png',mimeType:'image/png',buffer:Buffer.from('not an image')});
    await page.getByText('Could not decode a valid, bounded JPEG/PNG image', {exact: true}).waitFor();
    assert.deepEqual(errors, []);
    assert.deepEqual(externalRequests, []);
    console.log(JSON.stringify({result:'passed',initial_page_and_detection_ms:loadMs,one_slider_browser_round_trip_ms:interactiveMs,
      checks:['sample render','cached detection','slider','visible default displacement','face detail','wipe','difference','full portrait','overlay','reset','clear','invalid file','zero external browser requests','no browser exceptions']},null,2));
  } finally {await browser.close();}
})().catch(error => {console.error(error);process.exit(1);});
