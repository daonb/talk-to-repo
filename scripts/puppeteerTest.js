const puppeteer = require('puppeteer');

(async () => {
  const browser = await puppeteer.launch();
  const page = await browser.newPage();

  // Log the URL of every request made
  page.on('request', request => console.log(`Request URL: ${request.url()}`));

  // Log any errors or console log messages.
  page.on('console', message => {
    console.log(`Console message: ${message.type().substr(0, 3).toUpperCase()} ${message.text()}`);
  });

  await page.goto('http://localhost:3000');

  await page.waitForXPath('//button[contains(text(), "Load Repo")]');

  const buttons = await page.$x('//button[contains(text(), "Load Repo")]');
  await buttons[0].click();

  // Delay to allow the request to be processed
  await new Promise(resolve => setTimeout(resolve, 2000));

  await browser.close();
})();
