const {chromium}=require('playwright');
const assert=require('node:assert/strict');
(async()=>{const b=await chromium.launch({headless:true,executablePath:process.env.CHROMIUM_PATH||'/usr/bin/chromium'});try{
 const p=await b.newPage({viewport:{width:1500,height:1450}});const external=[],errors=[];p.on('pageerror',e=>errors.push(e.message));
 await p.route('**/*',r=>{const h=new URL(r.request().url()).hostname;if(h&&!['localhost','127.0.0.1'].includes(h)){external.push(h);return r.abort();}return r.continue();});
 await p.goto('http://127.0.0.1:8501');await p.getByText('Use public-domain NASA sample',{exact:true}).click();
 await p.getByText(/Local render:/).waitFor({timeout:60000});
 await p.locator('[data-testid=stImage] img').nth(1).evaluate(i=>i.decode());
 await p.screenshot({path:'experiments/geometry_renderer/screenshots/local-aging-ui.png',fullPage:true});
 await p.getByRole('combobox',{name:'Comparison view',exact:true}).click();await p.getByRole('option',{name:'Before / after wipe',exact:true}).click();await p.getByRole('heading',{name:'Aligned before / after comparison'}).waitFor();
 await p.getByRole('button',{name:'Reset aging',exact:true}).click();await p.getByText('Current and target age are equal: no additional aging is applied.',{exact:true}).waitFor();
 await p.getByRole('button',{name:'Clear image from this session',exact:true}).click();await p.getByText(/Select a local portrait with exactly one/).waitFor();
 assert.deepEqual(external,[]);assert.deepEqual(errors,[]);console.log('Passed: default full render, wipe, equal-age reset, clear, zero external requests, no browser errors.');
 }finally{await b.close();}})().catch(e=>{console.error(e);process.exit(1)});
