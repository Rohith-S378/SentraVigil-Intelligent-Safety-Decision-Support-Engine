const express = require('express');
const cors = require('cors');
const path = require('path');
const fs = require('fs');
const { spawn } = require('child_process');

const PORT_CENTRAL = 8000;
const PORT_MOBILE = 8001;
const API_TOKEN = process.env.TOURSHIELD_API_TOKEN || 'dev-only-change-me';
const EVALUATION_TIMEOUT_MS = 5000;

// Load config
let config = { latent_dim: 8, hidden_dim: 16, window_size: 20, reconstruction_threshold: 0.45 };
try {
  config = require('./best_config.json');
} catch (e) {
  console.log('Using default config');
}

// Global state
const BASE_LAT = 15.4989;
const BASE_LON = 73.8278;

// Persistent Database Layer for Location Secrecy Policy
const PRIVACY_DB_PATH = path.join(__dirname, 'privacy_db.json');
let privacyStore = {};

function loadPrivacyStore() {
  try {
    if (fs.existsSync(PRIVACY_DB_PATH)) {
      privacyStore = JSON.parse(fs.readFileSync(PRIVACY_DB_PATH, 'utf8'));
    }
  } catch (e) {
    console.error('Error reading privacy_db.json:', e);
  }
}

function savePrivacyStore() {
  try {
    fs.writeFileSync(PRIVACY_DB_PATH, JSON.stringify(privacyStore, null, 2), 'utf8');
  } catch (e) {
    console.error('Error saving privacy_db.json:', e);
  }
}

loadPrivacyStore();

const DISPATCH_LOG_PATH = path.join(__dirname, 'dispatch_log.json');
const GEOFENCES_PATH = path.join(__dirname, 'geofences.json');
function loadJsonArray(filePath) {
  try { return fs.existsSync(filePath) ? JSON.parse(fs.readFileSync(filePath, 'utf8')) : []; }
  catch (e) { console.error(`Error reading ${path.basename(filePath)}:`, e); return []; }
}
function saveJson(filePath, value) {
  try { fs.writeFileSync(filePath, JSON.stringify(value, null, 2), 'utf8'); }
  catch (e) { console.error(`Error saving ${path.basename(filePath)}:`, e); }
}

// System-wide Location Privacy Transformation Policy
function applyPrivacyPolicy(touristAgent, privacyState) {
  const isPrivate = privacyState && privacyState.enabled;

  if (!isPrivate) {
    return {
      latitude: touristAgent.lat,
      longitude: touristAgent.lon,
      isExact: true,
      privacyMode: 'EXACT',
      accuracyMeters: 5,
      banner: null
    };
  }

  // PRIVACY MODE: Quantize / snap coordinates to a 500-meter grid cell (~0.0045 deg)
  // plus offset to cell center so exact location cannot be reverse-engineered
  const GRID_SIZE = 0.005; // ~550m at 15 deg latitude
  const gridLat = Math.floor(touristAgent.lat / GRID_SIZE) * GRID_SIZE + (GRID_SIZE / 2);
  const gridLon = Math.floor(touristAgent.lon / GRID_SIZE) * GRID_SIZE + (GRID_SIZE / 2);

  return {
    latitude: parseFloat(gridLat.toFixed(4)),
    longitude: parseFloat(gridLon.toFixed(4)),
    isExact: false,
    privacyMode: 'APPROXIMATE_GEOCELL',
    accuracyMeters: 500,
    banner: '🔒 Location Protected (Approximate ~500m Grid Area)'
  };
}

const geofences = loadJsonArray(GEOFENCES_PATH); // Array of polygons { points: [[lat,lon],...], type: 'restricted' }
const customPaths = {}; // id -> [[lat,lon],...]
const handsetAlerts = {}; // id -> [{ message: '...', timestamp: ... }]

function isPointInPoly(poly, pt) {
  const x = pt[0], y = pt[1];
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const xi = poly[i][0], yi = poly[i][1];
    const xj = poly[j][0], yj = poly[j][1];
    const intersect = ((yi > y) !== (yj > y)) && (x < (xj - xi) * (y - yi) / (yj - yi) + xi);
    if (intersect) inside = !inside;
  }
  return inside;
}

class TouristAgent {
  constructor(id, name, initialOffset) {
    this.id = id;
    this.name = name;
    this.lat = BASE_LAT + initialOffset[0];
    this.lon = BASE_LON + initialOffset[1];
    this.heading = Math.random() * 360;
    this.speed = 1.4;
    this.journeyStart = Date.now();
    this.windowHistory = [];
    this.injectedAnomaly = null;
    this.anomalySteps = 0;
    this.latestEvaluation = { reconstruction_error: 0.0, is_anomaly: false, feature_breakdown: {} };
    this.pathIndex = 0;

    // Fill initial window
    for (let i = 0; i < config.window_size; i++) {
      this.stepSimulation(null);
    }
  }

  stepSimulation(anomalyMode) {
    if (anomalyMode !== undefined && anomalyMode !== null) {
      this.injectedAnomaly = anomalyMode;
      this.anomalySteps = 0;
    }

    const mode = this.injectedAnomaly;
    let speed, inactivity, deviation, directionChanges, zoneLevel;

    // Check Geofences
    zoneLevel = 0;
    for (const fence of geofences) {
      if (isPointInPoly(fence.points, [this.lat, this.lon])) {
        zoneLevel = 2; // Immediate Restricted Zone entry
        break;
      }
    }

    // Interactive Path Following Logic
    const customPath = customPaths[this.id];
    if (customPath && customPath.length > 0) {
      const target = customPath[this.pathIndex];
      const dx = target[0] - this.lat;
      const dy = target[1] - this.lon;
      const dist = Math.sqrt(dx * dx + dy * dy);
      
      speed = 1.4;
      inactivity = 0.0;
      deviation = 0.0; // Perfect tracking on custom path
      directionChanges = 0;
      
      if (dist < 0.0001) {
        this.pathIndex = (this.pathIndex + 1) % customPath.length;
      } else {
        const moveStep = 0.00001;
        this.lat += (dx / dist) * moveStep;
        this.lon += (dy / dist) * moveStep;
        this.heading = Math.atan2(dy, dx) * 180 / Math.PI;
      }
    } else if (mode === 'restricted_zone_inactivity') {
      speed = Math.max(0, 0.1 + Math.random() * 0.1);
      inactivity = (this.anomalySteps + 1) * 2.0;
      deviation = Math.min(45.0, 5.0 + this.anomalySteps * 1.5);
      directionChanges = Math.random() < 0.2 ? 1 : 0;
      zoneLevel = 2;
      this.lat += (Math.random() - 0.5) * 0.00002;
      this.lon += (Math.random() - 0.5) * 0.00002;
    } else if (mode === 'route_deviation') {
      speed = 1.6 + Math.random() * 0.3;
      inactivity = 0.0;
      deviation = Math.min(80.0, 10.0 + this.anomalySteps * 4.0);
      directionChanges = Math.random() < 0.3 ? 2 : 1;
      zoneLevel = Math.max(zoneLevel, deviation > 30 ? 1 : 0);
      this.heading += 15.0;
      const rad = (this.heading * Math.PI) / 180;
      this.lat += speed * Math.cos(rad) * 0.00001;
      this.lon += speed * Math.sin(rad) * 0.00001;
    } else if (mode === 'direction_burst') {
      speed = 2.0 + Math.random() * 0.4;
      inactivity = 0.0;
      deviation = Math.max(15.0, 12.0 + Math.random() * 15);
      directionChanges = Math.floor(Math.random() * 3) + 4;
      zoneLevel = Math.max(zoneLevel, 1);
      this.heading += (Math.random() < 0.5 ? 90 : -90);
      const rad = (this.heading * Math.PI) / 180;
      this.lat += speed * Math.cos(rad) * 0.00001;
      this.lon += speed * Math.sin(rad) * 0.00001;
    } else {
      // Normal walking
      speed = 1.2 + Math.random() * 0.4;
      inactivity = 0.0;
      deviation = Math.abs(2.0 + Math.random() * 2.5);
      directionChanges = Math.random() < 0.15 ? 1 : 0;
      this.heading += (Math.random() - 0.5) * 10;
      const rad = (this.heading * Math.PI) / 180;
      this.lat += speed * Math.cos(rad) * 0.000008;
      this.lon += speed * Math.sin(rad) * 0.000008;
    }

    if (this.injectedAnomaly) {
      this.anomalySteps++;
      if (this.anomalySteps > 25) {
        this.injectedAnomaly = null;
        this.anomalySteps = 0;
      }
    }

    const journeyDuration = (Date.now() - this.journeyStart) / 1000;
    const features = [
      parseFloat(speed.toFixed(2)),
      parseFloat(inactivity.toFixed(1)),
      parseFloat(deviation.toFixed(1)),
      parseFloat(directionChanges),
      parseFloat(zoneLevel),
      parseFloat(journeyDuration.toFixed(1))
    ];

    this.windowHistory.push(features);
    if (this.windowHistory.length > config.window_size) {
      this.windowHistory.shift();
    }
  }
}

const tourists = {
  'TS-101': new TouristAgent('TS-101', 'Aarav Sharma (VIP)', [0.001, 0.001]),
  'TS-102': new TouristAgent('TS-102', 'Elena Rostova', [-0.002, 0.002]),
  'TS-103': new TouristAgent('TS-103', 'Kenji Takahashi', [0.003, -0.001]),
  'TS-104': new TouristAgent('TS-104', 'Sophia Martinez', [-0.001, -0.003]),
  'TS-105': new TouristAgent('TS-105', 'David Miller', [0.002, 0.003])
};

const dispatches = loadJsonArray(DISPATCH_LOG_PATH);

// Correlated, bounded IPC to the trained ONNX inference bridge.
let pythonProcess = null;
let pythonRestarts = 0;
const evaluationCallbacks = new Map();
let pythonBuffer = '';
function rejectPendingEvaluations(error) {
  for (const pending of evaluationCallbacks.values()) {
    clearTimeout(pending.timeout);
    pending.callback(error, null);
  }
  evaluationCallbacks.clear();
}
function spawnPythonBridge() {
  const pythonBin = process.platform === 'win32' ? 'python' : 'python3';
  pythonProcess = spawn(pythonBin, ['-u', path.join(__dirname, 'backend', 'model_bridge.py')]);
  pythonBuffer = '';
  pythonProcess.stdout.on('data', (data) => {
  pythonBuffer += data.toString();
  let lines = pythonBuffer.split('\n');
  pythonBuffer = lines.pop(); // keep last incomplete line

  for (let line of lines) {
    if (!line.trim()) continue;
    try {
      const parsed = JSON.parse(line);
      const pending = evaluationCallbacks.get(parsed.tourist_id);
      if (!pending) continue;
      evaluationCallbacks.delete(parsed.tourist_id);
      clearTimeout(pending.timeout);
      if (parsed.error) pending.callback(new Error(parsed.error), null);
      else pending.callback(null, parsed);
    } catch (e) {
      console.error('Error parsing python bridge output:', e, 'Line:', line);
    }
  }
  });
  pythonProcess.stderr.on('data', (data) => {
  console.error('Python bridge stderr:', data.toString());
  });
  pythonProcess.stdin.on('error', (err) => rejectPendingEvaluations(new Error(`Python bridge stdin error: ${err.message}`)));
  pythonProcess.on('error', (err) => console.error('Python bridge spawn error:', err));
  pythonProcess.on('exit', (code) => {
    console.error(`Python bridge exited (code ${code}).`);
    rejectPendingEvaluations(new Error('Python bridge crashed'));
    if (pythonRestarts < 5) {
      pythonRestarts++;
      setTimeout(spawnPythonBridge, 1000 * pythonRestarts);
    } else console.error('Python bridge failed 5 times — manual restart required.');
  });
}
function evaluateWindowOnEdge(touristId, windowHistory, callback) {
  if (!pythonProcess || !pythonProcess.stdin.writable) return callback(new Error('Python bridge is unavailable'), null);
  const existing = evaluationCallbacks.get(touristId);
  if (existing) { clearTimeout(existing.timeout); existing.callback(new Error('Superseded by newer evaluation'), null); }
  const timeout = setTimeout(() => {
    if (evaluationCallbacks.get(touristId)?.timeout === timeout) {
      evaluationCallbacks.delete(touristId);
      callback(new Error('Python bridge evaluation timed out'), null);
    }
  }, EVALUATION_TIMEOUT_MS);
  evaluationCallbacks.set(touristId, { callback, timeout });
  try { pythonProcess.stdin.write(JSON.stringify({ tourist_id: touristId, window: windowHistory }) + '\n'); }
  catch (err) { clearTimeout(timeout); evaluationCallbacks.delete(touristId); callback(err, null); }
}
spawnPythonBridge();

// Background loop simulating edge evaluations over LTE (every 2s)
setInterval(() => {
  Object.keys(tourists).forEach(id => {
    const agent = tourists[id];
    agent.stepSimulation();
    evaluateWindowOnEdge(id, agent.windowHistory, (err, result) => {
      if (!err && result) {
        agent.latestEvaluation = result;
      }
    });
  });
}, 2000);

// --- Dual Express Servers ---
const appCentral = express();
const appMobile = express();

appCentral.use(cors());
appCentral.use(express.json());
appMobile.use(cors());
appMobile.use(express.json());
function requireAuth(req, res, next) {
  const token = req.headers.authorization?.replace(/^Bearer\s+/, '');
  if (token !== API_TOKEN) return res.status(401).json({ error: 'Unauthorized' });
  next();
}
function serveDashboard(filePath) {
  return (req, res) => {
    try {
      const html = fs.readFileSync(filePath, 'utf8').replace('%%TOURSHIELD_API_TOKEN%%', API_TOKEN);
      res.type('html').send(html);
    } catch (e) { res.status(500).send('Unable to load dashboard'); }
  };
}
appCentral.get('/', serveDashboard(path.join(__dirname, 'static', 'central', 'index.html')));
appMobile.get('/', serveDashboard(path.join(__dirname, 'static', 'mobile', 'index.html')));

// Serve static assets
appCentral.use(express.static(path.join(__dirname, 'static', 'central')));
appMobile.use(express.static(path.join(__dirname, 'static', 'mobile')));

// Sibling communication router
function getStates() {
  const data = {};
  Object.keys(tourists).forEach(id => {
    const agent = tourists[id];
    const privacyState = privacyStore[id] || { enabled: false, mode: 'EXACT', radiusMeters: 0 };
    const safeLocation = applyPrivacyPolicy(agent, privacyStore[id]);
    
    data[id] = {
      tourist_id: agent.id,
      name: agent.name,
      // Privacy-transformed coordinates (exact or grid-quantized)
      latitude: safeLocation.latitude,
      longitude: safeLocation.longitude,
      isExact: safeLocation.isExact,
      privacyMode: safeLocation.privacyMode,
      accuracyMeters: safeLocation.accuracyMeters,
      privacyBanner: safeLocation.banner,
      latest_features: {
        avg_speed: agent.windowHistory[agent.windowHistory.length - 1][0],
        inactivity_duration: agent.windowHistory[agent.windowHistory.length - 1][1],
        route_deviation: agent.windowHistory[agent.windowHistory.length - 1][2],
        direction_changes: agent.windowHistory[agent.windowHistory.length - 1][3],
        zone_level: agent.windowHistory[agent.windowHistory.length - 1][4],
        journey_duration: agent.windowHistory[agent.windowHistory.length - 1][5]
      },
      edge_evaluation: agent.latestEvaluation,
      injected_anomaly: agent.injectedAnomaly,
      privacy_mode: privacyState.enabled,
      privacyState: privacyState
    };
  });
  return data;
}

// API: Tourists
const touristHandler = (req, res) => res.json(getStates());
appCentral.get('/api/tourists', touristHandler);
appMobile.get('/api/tourists', touristHandler);

// API: Trigger Anomaly / Custom Feature Injection
const triggerAnomalyHandler = (req, res) => {
  const { tourist_id, anomaly_type, custom_features } = req.body;
  const agent = tourists[tourist_id];
  if (agent) {
    if (custom_features) {
      // Direct custom feature vector injection
      const journeyDuration = (Date.now() - agent.journeyStart) / 1000;
      const feat = [
        parseFloat(custom_features.speed || 1.4),
        parseFloat(custom_features.inactivity || 0),
        parseFloat(custom_features.deviation || 0),
        parseFloat(custom_features.direction_changes || 0),
        parseFloat(custom_features.zone_level || 0),
        parseFloat(journeyDuration.toFixed(1))
      ];
      // Push multiple copies to fill window for custom evaluation
      agent.injectedAnomaly = 'custom';
      agent.anomalySteps = 1;
      for (let i = 0; i < 5; i++) {
        agent.windowHistory.push(feat);
        if (agent.windowHistory.length > config.window_size) agent.windowHistory.shift();
      }
    } else {
      agent.stepSimulation(anomaly_type);
    }

    evaluateWindowOnEdge(tourist_id, agent.windowHistory, (err, result) => {
      if (!err && result) {
        agent.latestEvaluation = result;
        res.json({ status: 'SUCCESS', data: getStates()[tourist_id] });
      } else {
        res.status(500).json({ error: 'Evaluation failed' });
      }
    });
  } else {
    res.status(404).json({ error: 'Tourist not found' });
  }
};
appCentral.post('/api/trigger_anomaly', requireAuth, triggerAnomalyHandler);
appMobile.post('/api/trigger_anomaly', requireAuth, triggerAnomalyHandler);

// API: Gemini LLM Tactical Layer with fallback
const https = require('https');
const geminiHandler = (req, res) => {
  const { tourist_id, anomaly_data, location_name } = req.body;
  const apiKey = process.env.GEMINI_API_KEY || '';

  const zone = Math.round(anomaly_data.feature_breakdown?.zone_level?.latest_val || 0);
  const speed = parseFloat((anomaly_data.feature_breakdown?.avg_speed?.latest_val || 0).toFixed(1));
  const inactivity = Math.round(anomaly_data.feature_breakdown?.inactivity_duration?.latest_val || 0);
  const deviation = Math.round(anomaly_data.feature_breakdown?.route_deviation?.latest_val || 0);
  const turn_burst = Math.round(anomaly_data.feature_breakdown?.direction_changes?.latest_val || 0);

  // Static/Fallback analytical engine with clean rounding
  let fallback = {
    risk_assessment: `Tourist ${tourist_id} has exceeded normal trajectory parameters inside Goa Coastal Sector. Path deviation of ${deviation}m indicates possible disorientation or trail loss.`,
    primary_threat_type: 'DISORIENTATION_PANIC_FLIGHT',
    recommended_decision: 'BROADCAST EDGE WAYFINDING & STAGE QRT SQUAD',
    action_steps: [
      'Publish interactive high-resolution route map to tourist handset.',
      'Notify nearest beach lifesaver tower (Tower 4).',
      'Monitor secondary telemetry loop for path re-alignment.'
    ],
    police_dispatch_message: `[TOURSHIELD SEARCH ADVISORY] Tourist ${tourist_id} is disoriented near cliff boundary. Last speed ${speed} m/s, Deviation ${deviation}m. Ready QRT interception.`
  };

  if (zone === 2 || inactivity > 60) {
    fallback.primary_threat_type = 'CRITICAL_INCAPACITATION';
    fallback.risk_assessment = `Tourist ${tourist_id} is stationary (${inactivity}s) in a designated high-risk Restricted Danger Zone. Highly likely fall, medical trauma, or extreme environmental hazard.`;
    fallback.recommended_decision = 'IMMEDIATE CO-LOCATED POLICE & CLIFF RESCUE DISPATCH';
    fallback.action_steps = [
      'Instantly stream real-time coordinate package to nearest Tactical patrol.',
      'Siren trigger on nearest coastal speaker node.',
      'Activate VIP device edge audio transceiver.'
    ];
    fallback.police_dispatch_message = `[TOURSHIELD PRIORITY DISPATCH] Tourist ${tourist_id} INCAPACITATED at Cliff perimeter. Inactivity: ${inactivity}s, Zone: Restricted. Prompt response required.`;
  }

    if (apiKey) {
      const cleanSpeed = parseFloat(speed.toFixed(1));
      const cleanDeviation = Math.round(deviation);
      const prompt = `You are TourShield Command AI. Provide a JSON response evaluating tourist ${tourist_id} anomaly. Metrics: speed=${cleanSpeed}, inactivity=${inactivity}, deviation=${cleanDeviation}, zone=${zone}. Format: {"risk_assessment": "...", "primary_threat_type": "...", "recommended_decision": "...", "action_steps": ["step1", "step2"], "police_dispatch_message": "..."}`;
      const postData = JSON.stringify({
        contents: [{ parts: [{ text: prompt }] }],
        generationConfig: { response_mime_type: 'application/json' }
      });

    const options = {
      hostname: 'generativelanguage.googleapis.com',
      port: 443,
      path: `/v1beta/models/gemini-2.0-flash:generateContent?key=${apiKey}`,
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Content-Length': Buffer.byteLength(postData)
      }
    };

    const reqGemini = https.request(options, (resGemini) => {
      let body = '';
      resGemini.on('data', chunk => body += chunk);
      resGemini.on('end', () => {
        try {
          const parsed = JSON.parse(body);
          let text = parsed.candidates[0].content.parts[0].text;
          // Sanitization for markdown code block formatting
          text = text.replace(/```json/gi, '').replace(/```/g, '').trim();
          res.json(JSON.parse(text));
        } catch (e) {
          console.error('Gemini parse error, using fallback:', e);
          res.json({ ...fallback, llm_engine: 'Gemini Hybrid Fallback Matrix' });
        }
      });
    });

    reqGemini.on('error', () => res.json({ ...fallback, llm_engine: 'Gemini Hybrid Fallback Matrix' }));
    reqGemini.write(postData);
    reqGemini.end();
  } else {
    res.json({ ...fallback, llm_engine: 'TourShield Fallback Analytical Matrix' });
  }
};
appCentral.post('/api/gemini_decision', geminiHandler);
appMobile.post('/api/gemini_decision', geminiHandler);

// API: Notify Police
const notifyPoliceHandler = (req, res) => {
  const { tourist_id, location, threat_type, decision, message } = req.body;
  const record = {
    dispatch_id: `POL-DISPATCH-${dispatches.length + 101}`,
    tourist_id,
    location,
    threat_type,
    decision,
    message,
    status: 'EN_ROUTE',
    eta_minutes: Math.floor(Math.random() * 3) + 3,
    responder_unit: 'Sector 4 Coastal Tactical Patrol (Unit 4B)',
    timestamp: new Date().toLocaleTimeString()
  };
  dispatches.unshift(record);
  saveJson(DISPATCH_LOG_PATH, dispatches);
  res.json({ status: 'POLICE_DISPATCHED', record });
};
appCentral.post('/api/notify_police', requireAuth, notifyPoliceHandler);
appMobile.post('/api/notify_police', requireAuth, notifyPoliceHandler);

// API: Dispatch History
const dispatchesHandler = (req, res) => res.json(dispatches);
appCentral.get('/api/police_dispatches', dispatchesHandler);
appMobile.get('/api/police_dispatches', dispatchesHandler);

// --- New Bidirectional & Drawing APIs ---

appCentral.post('/api/send_handset_alert', requireAuth, (req, res) => {
  const { tourist_id, message } = req.body;
  if (!handsetAlerts[tourist_id]) handsetAlerts[tourist_id] = [];
  handsetAlerts[tourist_id].push({ message, timestamp: Date.now() });
  res.json({ status: 'ALERT_QUEUED' });
});

appCentral.post('/api/alert_all_in_danger', requireAuth, (req, res) => {
  const { message } = req.body;
  const alerted = [];
  Object.keys(tourists).forEach(id => {
    const agent = tourists[id];
    if (agent.latestEvaluation && agent.latestEvaluation.is_anomaly) {
      if (!handsetAlerts[id]) handsetAlerts[id] = [];
      handsetAlerts[id].push({ message: message || '🚨 CRITICAL SECURITY WARNING: Stay in safe zones. Assistance dispatched.', timestamp: Date.now() });
      alerted.push(id);
    }
  });
  res.json({ status: 'BROADCAST_SENT', alerted_tourists: alerted });
});

appMobile.get('/api/handset_alerts/:id', (req, res) => {
  const id = req.params.id;
  const alerts = handsetAlerts[id] || [];
  handsetAlerts[id] = []; // Clear after delivery
  res.json(alerts);
});

appCentral.post('/api/toggle_privacy', requireAuth, (req, res) => {
  const { tourist_id, enabled } = req.body;
  if (!privacyStore[tourist_id]) {
    privacyStore[tourist_id] = { enabled: false, mode: 'EXACT', radiusMeters: 0 };
  }
  privacyStore[tourist_id].enabled = !!enabled;
  privacyStore[tourist_id].updatedAt = new Date().toISOString();
  savePrivacyStore();
  console.log(`[PRIVACY] Tourist ${tourist_id} privacy set to: ${enabled}`);
  res.json({ status: 'PRIVACY_TOGGLED', tourist_id, enabled: privacyStore[tourist_id].enabled });
});
appMobile.post('/api/toggle_privacy', requireAuth, (req, res) => {
  const { tourist_id, enabled } = req.body;
  if (!privacyStore[tourist_id]) privacyStore[tourist_id] = { enabled: false, mode: 'EXACT', radiusMeters: 0 };
  privacyStore[tourist_id].enabled = !!enabled;
  privacyStore[tourist_id].updatedAt = new Date().toISOString();
  savePrivacyStore();
  res.json({ status: 'PRIVACY_TOGGLED', tourist_id, enabled: privacyStore[tourist_id].enabled });
});

// GET endpoint for handsets to sync their own privacy status on load
appMobile.get('/api/privacy_status/:id', (req, res) => {
  const id = req.params.id;
  const state = privacyStore[id] || { enabled: false, mode: 'EXACT', radiusMeters: 0 };
  res.json({ enabled: state.enabled });
});

appCentral.post('/api/update_geofences', requireAuth, (req, res) => {
  const { points } = req.body;
  geofences.push({ points, type: 'restricted' });
  saveJson(GEOFENCES_PATH, geofences);
  res.json({ status: 'GEOFENCE_ADDED', count: geofences.length });
});

appCentral.post('/api/update_path', requireAuth, (req, res) => {
  const { tourist_id, points } = req.body;
  const privacyState = privacyStore[tourist_id] || { enabled: false };
  if (privacyState.enabled) {
    return res.status(403).json({ 
      status: 'FORBIDDEN', 
      message: `Cannot assign route: ${tourist_id} has Location Secrecy ENABLED. Disable Location Secrecy to assign custom routes.`
    });
  }
  customPaths[tourist_id] = points;
  if (tourists[tourist_id]) tourists[tourist_id].pathIndex = 0;
  res.json({ status: 'PATH_UPDATED' });
});

appCentral.post('/api/clear_drawings', requireAuth, (req, res) => {
  geofences.length = 0;
  Object.keys(customPaths).forEach(k => delete customPaths[k]);
  saveJson(GEOFENCES_PATH, geofences);
  res.json({ status: 'CLEARED' });
});

// Start Central Command dashboard
appCentral.listen(PORT_CENTRAL, '127.0.0.1', () => {
  console.log(`\n======================================================`);
  console.log(`[SERVICE A] Central Intelligence Center is active!`);
  console.log(`URL: http://127.0.0.1:${PORT_CENTRAL}`);
  console.log(`======================================================`);
});

// Start Mobile App simulator
appMobile.listen(PORT_MOBILE, '127.0.0.1', () => {
  console.log(`[SERVICE B] Mobile App Edge Simulator is active!`);
  console.log(`URL: http://127.0.0.1:${PORT_MOBILE}`);
  console.log(`======================================================\n`);
});
