// Imprime el HTML del documento a PDF con Chromium (Playwright).
const { chromium } = require('playwright');
(async () => {
  const [entrada, salida] = process.argv.slice(2);
  const navegador = await chromium.launch();
  const pagina = await navegador.newPage();
  await pagina.goto('file://' + entrada, { waitUntil: 'networkidle' });
  await pagina.pdf({
    path: salida, format: 'A4', printBackground: true, preferCSSPageSize: true,
    displayHeaderFooter: true,
    headerTemplate: '<div></div>',
    footerTemplate: '<div style="font-size:8pt;color:#78909C;width:100%;text-align:center;">' +
      'RZI - Documento de estudio | pagina <span class="pageNumber"></span> de <span class="totalPages"></span></div>',
  });
  await navegador.close();
})();
