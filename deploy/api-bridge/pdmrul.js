// Bridge: pdm-xgboost (FastAPI, 127.0.0.1:8000) -> dashboard iothub.
// Read-only relay of the RUL predictions the pdm-predict@ timers already log. It never calls
// /predict (that one scores, writes sqlite and can send Telegram) -- only /rul/*.
//
// Install on the IoT server:
//   cp pdmrul.js /opt/api/routes/pdmrul.js
//   index.js:  const pdmrul = require('./routes/pdmrul');
//              app.use('/api/pdmrul', pdmrul);
//
// Endpoints (GET):
//   /api/pdmrul/:iddev/latest
//   /api/pdmrul/:iddev/history?sensor=&subtype=&since=2026-10-01T00:00:00&limit=500

const express = require('express');
const axios = require('axios');

const router = express.Router();

const XGBOOST_URL = process.env.PDM_XGBOOST_URL || 'http://127.0.0.1:8000';
const HISTORY_PARAMS = ['sensor', 'subtype', 'since', 'limit'];

async function relay(res, path, params) {
  try {
    const r = await axios.get(`${XGBOOST_URL}${path}`, { params, timeout: 15000 });
    return res.status(200).json(r.data);
  } catch (error) {
    if (error.response) {
      // FastAPI answered with an error (404 unknown iddev, 422 bad param, ...): pass it through
      return res.status(error.response.status).json(error.response.data);
    }
    console.error('pdmrul bridge error:', error.message);
    return res.status(502).json({
      error: 'pdm-xgboost service unreachable',
      detail: error.message
    });
  }
}

router.get('/:iddev/latest', (req, res) => {
  const iddev = parseInt(req.params.iddev, 10);
  if (Number.isNaN(iddev)) {
    return res.status(400).json({ error: 'iddev must be an integer' });
  }
  return relay(res, `/rul/${iddev}/latest`);
});

router.get('/:iddev/history', (req, res) => {
  const iddev = parseInt(req.params.iddev, 10);
  if (Number.isNaN(iddev)) {
    return res.status(400).json({ error: 'iddev must be an integer' });
  }
  const params = {};
  for (const key of HISTORY_PARAMS) {
    if (req.query[key] !== undefined) params[key] = req.query[key];
  }
  return relay(res, `/rul/${iddev}/history`, params);
});

module.exports = router;
