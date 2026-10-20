# Deploying the human-participant portal

The portal (`index.html`) links the three task pages. Each task page stores its trial log in
the participant's browser (`localStorage`); when all three tasks are complete, the portal
sends one JSON payload to a webhook and redirects the participant to the recruitment
platform's completion page.

## 1. Host the static files

Any static web host works (e.g. GitHub Pages). For a local test:

```bash
cd experiments/web/human
python -m http.server 8000      # then open http://localhost:8000/
```

## 2. Create the data webhook (Google Apps Script)

1. Create a Google Sheet, open **Extensions → Apps Script** and paste:

```javascript
function doPost(e) {
  var ss = SpreadsheetApp.getActiveSpreadsheet();
  var body = JSON.parse(e.postData.contents);
  var stamp = body.timestamp || new Date().toISOString();
  // The portal sends {experimentId: "ALL", subjectId, timestamp, dnc, fip, ctd};
  // each task field holds that task's JSON log as a string.
  ["dnc", "fip", "ctd"].forEach(function (task) {
    if (!body[task]) return;
    var name = task.toUpperCase();
    var sheet = ss.getSheetByName(name) || ss.insertSheet(name);
    if (sheet.getLastRow() === 0) sheet.appendRow(["Timestamp", "Subject ID", "Experiment ID", "Data (JSON)"]);
    sheet.appendRow([stamp, body.subjectId, name, body[task]]);
  });
  return ContentService.createTextOutput(JSON.stringify({status: "success"}))
    .setMimeType(ContentService.MimeType.JSON);
}
```

2. **Deploy → New deployment → Web app**, execute as *Me*, access *Anyone*; copy the Web app URL.
3. In `index.html`, replace `YOUR_WEBHOOK_URL_HERE` with that URL. The request uses
   `mode: "no-cors"`, so the browser does not read the response.

## 3. Recruitment platform

Replace `YOUR_PROLIFIC_COMPLETION_CODE` in `index.html` (four occurrences) with your
study's completion code, or point the redirect to your own platform.

## 4. Retrieving data

Each sheet (`DNC`, `FIP`, `CTD`) holds one row per participant with the task log in the last
column; export to CSV and parse that JSON column (avoid XLSX, which truncates cells longer than
32,767 characters). In the released human dataset the same task logs appear under the source
labels `DSB` (= DNC) and `TPB` (= CTD).
