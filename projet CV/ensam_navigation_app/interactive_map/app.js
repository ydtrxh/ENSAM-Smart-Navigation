// ═══════════════════════════════════════════════════════════════════════════
//  ENSAM Campus Interactive Map — app.js
// ═══════════════════════════════════════════════════════════════════════════

// ── 1. Category Colors & Building-specific overrides ─────────────────────

const CATEGORY_COLORS = {
    'Sports Facilities':    '#16a34a',
    'Residential':          '#64748b',
    'Academic Departments': '#7c3aed',
    'Academic Services':    '#0d9488',
    'Administration':       '#6b7280',
    'Libraries':            '#ea580c',
    'Amphitheaters':        '#0891b2',
    'Laboratories':         '#a855f7',
    'Student Services':     '#ec4899',
};

// Fine-grained per-building color (match screenshot palette)
const BUILDING_COLORS = {
    B01: '#5c6370', B02: '#4b5563', B03: '#a855f7', B04: '#7c3aed',
    B05: '#0891b2', B06: '#6d28d9', B07: '#6d28d9', B08: '#6d28d9',
    B09: '#059669', B10: '#7c3aed', B11: '#eab308', B12: '#8b5cf6',
    B13: '#3b82f6', B14: '#ea580c', B15: '#7c3aed', B16: '#3b82f6',
    B17: '#3b82f6', B18: '#64748b', B19: '#0891b2', B20: '#64748b',
    B21: '#64748b', B22: '#64748b', B23: '#059669', B24: '#7c3aed',
    B25: '#16a34a', B26: '#3b82f6', B27: '#f97316', B28: '#06b6d4',
    B29: '#4b5563', B30: '#1e40af',
};

function getColor(feature) {
    const id = feature.properties.building_id;
    if (BUILDING_COLORS[id]) return BUILDING_COLORS[id];
    return CATEGORY_COLORS[feature.properties.category] || '#4facfe';
}

// ── 2. Campus Road Network (pixel coords matching GeoJSON system) ────────

const CAMPUS_ROADS = [
    // Main entrance road  →  up to Admin I
    [[222, 11], [222, 72], [222, 150], [222, 189]],
    // Admin I  →  Forum  →  Admin II
    [[222, 189], [222, 261], [222, 283], [194, 283], [156, 295]],
    [[222, 283], [328, 283], [406, 283]],
    // Admin II east  →  TD rooms  →  Amphi 250
    [[406, 283], [500, 283], [589, 283], [639, 260], [692, 240], [744, 236]],
    // Admin north  →  Bibliothèque  →  Centre Langues
    [[206, 372], [206, 436], [183, 460], [141, 490], [98, 506]],
    // Bibliothèque  →  Amphi 450
    [[206, 436], [211, 470], [211, 517], [211, 576]],
    // Amphi 450  →  Departments
    [[278, 517], [367, 510], [367, 500]],
    // Department corridor (south)
    [[367, 580], [367, 500], [367, 439], [411, 400], [411, 356]],
    // Department east  →  Dept AE  →  Lab Civil
    [[550, 500], [550, 460], [706, 450], [706, 436]],
    // Lab Civil  →  Centre Recherche  →  GMS
    [[706, 436], [739, 394], [739, 358], [889, 358]],
    [[889, 358], [950, 350], [950, 322]],
    [[950, 372], [1194, 372], [1194, 322]],
    // GMS  →  Sports courts
    [[1044, 400], [1044, 467], [1111, 467]],
    [[1111, 467], [1178, 467], [1244, 467]],
    // To Football
    [[1244, 467], [1300, 467], [1370, 467]],
    // Buvette  →  Energétique
    [[1000, 464], [1012, 500], [1012, 522]],
    // North area: Mosque  →  Salle Conférence  →  Complexe Sportif
    [[183, 628], [183, 717], [283, 717]],
    [[283, 700], [428, 748], [661, 748]],
    [[661, 748], [661, 780], [800, 780], [800, 733]],
    // Complexe sportif  →  Dept Matériaux  →  Volley
    [[756, 748], [951, 748], [951, 700], [951, 650]],
    [[951, 650], [1100, 650], [1161, 650]],
    [[1161, 680], [1338, 680]],
    // Parking road
    [[500, 233], [500, 200], [500, 172], [533, 144], [533, 100]],
    [[533, 100], [667, 100]],
    // Atelier connector
    [[478, 210], [639, 210]],
    // Sports  →  Volley connector
    [[1111, 533], [1111, 600], [1100, 660]],
    // Bibliothèque to Admin II
    [[206, 372], [222, 344]],
    // Centre langues  →  Mosque
    [[141, 506], [141, 560], [183, 628]],
];

// ── 3. Building Marker SVG icon ──────────────────────────────────────────

const MARKER_SVG = `<svg viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 21h18"/><path d="M5 21V7l8-4v18"/><path d="M19 21V11l-6-4"/><path d="M9 9v.01"/><path d="M9 12v.01"/><path d="M9 15v.01"/><path d="M9 18v.01"/></svg>`;

// ── 4. Map Initialisation ────────────────────────────────────────────────

const IMG_W = 1600;
const IMG_H = 900;
const bounds = [[0, 0], [IMG_H, IMG_W]];

const map = L.map('map', {
    crs: L.CRS.Simple,
    minZoom: -1.5,
    maxZoom: 3,
    zoomControl: false,
    attributionControl: false,
});

L.control.zoom({ position: 'topright' }).addTo(map);

// Add flattened campus image
L.imageOverlay('assets/ensam_plan_flat.jpg', bounds, { className: 'dark-overlay' }).addTo(map);

map.fitBounds(bounds, { padding: [20, 20] });

// ── 5. State ─────────────────────────────────────────────────────────────

let geojsonLayer;
const allFeatures = [];           // { name, category, color, layer, feature }
const categoryVisibility = {};    // category → bool

// ── 6. Draw Roads ────────────────────────────────────────────────────────

function drawRoads() {
    CAMPUS_ROADS.forEach(coords => {
        const latlngs = coords.map(([x, y]) => L.latLng(y, x));
        L.polyline(latlngs, {
            color: 'rgba(255,255,255,0.22)',
            weight: 3.5,
            lineJoin: 'round',
            lineCap: 'round',
            interactive: false,
        }).addTo(map);
    });
}
drawRoads();

// ── 7. Styles ────────────────────────────────────────────────────────────

function getStyle(feature) {
    const color = getColor(feature);
    return {
        fillColor: color,
        weight: 1.5,
        opacity: .8,
        color: '#ffffffcc',
        fillOpacity: .55,
    };
}

function getHighlightStyle(feature) {
    const color = getColor(feature);
    return {
        fillColor: color,
        weight: 3,
        opacity: 1,
        color: '#fff',
        fillOpacity: .88,
    };
}

// ── 8. Popup Content ─────────────────────────────────────────────────────

function popupHTML(props) {
    const name  = props.name || 'N/A';
    const cat   = props.category || '';
    const bid   = props.building_id || '';
    const desc  = props.description || '';
    const color = BUILDING_COLORS[bid] || CATEGORY_COLORS[cat] || '#4facfe';
    return `
        <div class="popup-header" style="background:${color}">
            <div class="popup-title">${name}</div>
            <div class="popup-cat">${cat}${bid ? ' · ' + bid : ''}</div>
        </div>
        <div class="popup-body">
            <p class="popup-desc">${desc}</p>
        </div>`;
}

// ── 9. Info Panel Content ────────────────────────────────────────────────

function showInfoPanel(props) {
    const bid   = props.building_id || '';
    const name  = props.name || 'N/A';
    const cat   = props.category || '';
    const desc  = props.description || '';
    const type  = props.type || '';
    const cap   = props.capacity;
    const floors = props.floor_count;
    const color = BUILDING_COLORS[bid] || CATEGORY_COLORS[cat] || '#4facfe';

    let html = `
        <div class="info-header-band" style="background:linear-gradient(135deg, ${color}, ${color}dd)">
            <div class="info-building-name">${name}</div>
            <div class="info-building-id">${bid}${cat ? ' · ' + cat : ''}</div>
        </div>
        <div class="info-body">
            <div class="info-meta-row">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2"/><path d="M3 9h18"/></svg>
                <span class="info-meta-label">Type</span>
                <span class="info-meta-value">${type.replace(/_/g, ' ')}</span>
            </div>`;
    if (floors != null) {
        html += `
            <div class="info-meta-row">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 21h18"/><path d="M5 21V7l8-4v18"/><path d="M19 21V11l-6-4"/></svg>
                <span class="info-meta-label">Étages</span>
                <span class="info-meta-value">${floors}</span>
            </div>`;
    }
    if (cap) {
        html += `
            <div class="info-meta-row">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M23 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/></svg>
                <span class="info-meta-label">Capacité</span>
                <span class="info-meta-value">${cap}</span>
            </div>`;
    }
    html += `<div class="info-desc">${desc}</div></div>`;

    const panel = document.getElementById('infoPanel');
    document.getElementById('infoPanelContent').innerHTML = html;
    panel.classList.remove('hidden');
}

// ── 10. Per-feature Interactions ─────────────────────────────────────────

function onEachFeature(feature, layer) {
    const props = feature.properties;
    const name  = props.name || 'Unknown';
    const cat   = props.category || '';
    const bid   = props.building_id || '';
    const color = getColor(feature);

    allFeatures.push({ name, category: cat, color, layer, feature, bid });

    // Popup
    layer.bindPopup(popupHTML(props), {
        closeButton: true,
        className: 'custom-popup',
        minWidth: 240,
        maxWidth: 320,
    });

    // Blue circle marker at centroid
    const center = layer.getBounds().getCenter();
    const marker = L.marker(center, {
        icon: L.divIcon({
            className: '',
            html: `<div class="building-marker" style="background: ${color}">${MARKER_SVG}</div>`,
            iconSize: [28, 28],
            iconAnchor: [14, 14],
        }),
        interactive: true,
    }).addTo(map);

    marker.bindPopup(popupHTML(props), { className: 'custom-popup', minWidth: 240 });
    marker.on('click', () => {
        map.flyToBounds(layer.getBounds(), { padding: [80, 80], duration: .6 });
        showInfoPanel(props);
    });

    // Building name label (offset below marker)
    const labelText = (props.name || '').replace(/Département\s*/i, 'Dept. ').replace(/Laboratoires?\s*/i, 'Lab. ');
    const area = getBoundsArea(layer.getBounds());
    const isLarge = area > 25000;
    L.marker(L.latLng(center.lat - 18, center.lng), {
        icon: L.divIcon({
            className: `building-label${isLarge ? ' large' : ''}`,
            html: labelText,
            iconSize: [120, 20],
            iconAnchor: [60, 0],
        }),
        interactive: false,
    }).addTo(map);

    // Polygon interactions
    layer.on({
        mouseover(e) {
            e.target.setStyle(getHighlightStyle(feature));
            e.target.bringToFront();
        },
        mouseout(e) {
            if (!e.target.isPopupOpen()) geojsonLayer.resetStyle(e.target);
        },
        click(e) {
            map.flyToBounds(e.target.getBounds(), { padding: [80, 80], duration: .6 });
            showInfoPanel(props);
        },
        popupclose(e) {
            geojsonLayer.resetStyle(e.target);
        },
    });
}

function getBoundsArea(bounds) {
    const sw = bounds.getSouthWest();
    const ne = bounds.getNorthEast();
    return (ne.lng - sw.lng) * (ne.lat - sw.lat);
}

// ── 11. Load GeoJSON ─────────────────────────────────────────────────────

fetch('data/campus.geojson')
    .then(r => { if (!r.ok) throw new Error(r.status); return r.json(); })
    .then(data => {
        geojsonLayer = L.geoJSON(data, {
            style: getStyle,
            onEachFeature,
            coordsToLatLng: coords => L.latLng(coords[1], coords[0]),
        }).addTo(map);

        // Update building counter
        const counter = document.getElementById('counterNumber');
        counter.textContent = data.features.length;
        counter.classList.add('counter-animate');

        // Build legend
        buildLegend(data.features);
    })
    .catch(err => console.error('GeoJSON load error:', err));

// ── 12. Legend ───────────────────────────────────────────────────────────

function buildLegend(features) {
    const counts = {};
    features.forEach(f => {
        const cat = f.properties.category || 'Other';
        counts[cat] = (counts[cat] || 0) + 1;
        if (!(cat in categoryVisibility)) categoryVisibility[cat] = true;
    });

    const container = document.getElementById('legendContent');
    container.innerHTML = '';

    Object.entries(counts)
        .sort((a, b) => b[1] - a[1])
        .forEach(([cat, count]) => {
            const color = CATEGORY_COLORS[cat] || '#4facfe';
            const item = document.createElement('div');
            item.className = 'legend-item';
            item.innerHTML = `
                <span class="legend-swatch" style="background:${color}"></span>
                <span class="legend-label">${cat}</span>
                <span class="legend-count">${count}</span>`;
            item.addEventListener('click', () => toggleCategory(cat, item));
            container.appendChild(item);
        });
}

function toggleCategory(cat, itemEl) {
    categoryVisibility[cat] = !categoryVisibility[cat];
    itemEl.classList.toggle('dimmed', !categoryVisibility[cat]);

    geojsonLayer.eachLayer(layer => {
        const fcat = layer.feature.properties.category;
        if (fcat === cat) {
            if (categoryVisibility[cat]) {
                layer.setStyle(getStyle(layer.feature));
                layer.getElement && layer.getElement().style.setProperty('display', '');
            } else {
                layer.setStyle({ fillOpacity: 0, opacity: 0, weight: 0 });
            }
        }
    });
}

// Legend toggle
document.getElementById('legendToggle').addEventListener('click', function () {
    const panel = document.getElementById('legendPanel');
    const isHidden = panel.classList.contains('hidden');
    panel.classList.toggle('hidden', !isHidden);
    this.classList.toggle('active', isHidden);
});
document.getElementById('legendClose').addEventListener('click', () => {
    document.getElementById('legendPanel').classList.add('hidden');
    document.getElementById('legendToggle').classList.remove('active');
});

// ── 13. Info Panel Close ─────────────────────────────────────────────────

document.getElementById('infoPanelClose').addEventListener('click', () => {
    document.getElementById('infoPanel').classList.add('hidden');
});

// ── 14. Reset View ───────────────────────────────────────────────────────

document.getElementById('resetView').addEventListener('click', () => {
    map.flyToBounds(bounds, { padding: [20, 20], duration: .7 });
    document.getElementById('infoPanel').classList.add('hidden');
});

// ── 15. Search ───────────────────────────────────────────────────────────

const searchInput   = document.getElementById('searchInput');
const searchResults = document.getElementById('searchResults');

searchInput.addEventListener('input', function () {
    const q = this.value.toLowerCase().trim();
    if (q.length < 2) {
        searchResults.style.display = 'none';
        searchResults.innerHTML = '';
        return;
    }

    const matches = allFeatures.filter(f =>
        f.name.toLowerCase().includes(q) ||
        f.category.toLowerCase().includes(q) ||
        f.bid.toLowerCase().includes(q)
    );

    if (matches.length > 0) {
        searchResults.innerHTML = matches.map(m => `
            <div class="search-result-item" data-name="${m.name}">
                <span class="result-color" style="background:${m.color}"></span>
                <div>
                    <div class="result-name">${highlightMatch(m.name, q)}</div>
                    <div class="result-cat">${m.category}</div>
                </div>
            </div>`).join('');
    } else {
        searchResults.innerHTML = '<div class="search-result-item"><div class="result-name" style="color:var(--text-muted)">Aucun résultat trouvé</div></div>';
    }
    searchResults.style.display = 'block';
});

function highlightMatch(text, query) {
    const idx = text.toLowerCase().indexOf(query);
    if (idx < 0) return text;
    return text.slice(0, idx) + '<strong style="color:#a78bfa">' + text.slice(idx, idx + query.length) + '</strong>' + text.slice(idx + query.length);
}

searchResults.addEventListener('click', function (e) {
    const item = e.target.closest('.search-result-item');
    if (!item) return;
    const name = item.getAttribute('data-name');
    if (!name) return;

    const f = allFeatures.find(x => x.name === name);
    if (f) {
        f.layer.openPopup();
        map.flyToBounds(f.layer.getBounds(), { padding: [100, 100], duration: .7 });
        showInfoPanel(f.feature.properties);
        searchResults.style.display = 'none';
        searchInput.value = '';
    }
});

document.addEventListener('click', e => {
    if (!e.target.closest('#searchContainer')) {
        searchResults.style.display = 'none';
    }
});

// ── 16. Keyboard shortcut: Escape closes panels ─────────────────────────

document.addEventListener('keydown', e => {
    if (e.key === 'Escape') {
        document.getElementById('infoPanel').classList.add('hidden');
        document.getElementById('legendPanel').classList.add('hidden');
        document.getElementById('legendToggle').classList.remove('active');
        searchResults.style.display = 'none';
        searchInput.blur();
    }
});
