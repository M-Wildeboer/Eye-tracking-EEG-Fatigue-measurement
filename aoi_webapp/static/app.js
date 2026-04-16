const state = {
  images: [],
  currentImage: null,
  aois: [],
  selectedUid: null,
  isDrawing: false,
  isDragging: false,
  drawStart: null,
  tempRect: null,
  dragData: null,
  saveTimer: null,
};

const imageListEl = document.getElementById('imageList');
const aoiListEl = document.getElementById('aoiList');
const refreshImagesBtn = document.getElementById('refreshImagesBtn');
const aoiNameInput = document.getElementById('aoiNameInput');
const currentImageLabel = document.getElementById('currentImageLabel');
const saveStatus = document.getElementById('saveStatus');
const emptyState = document.getElementById('emptyState');
const viewerStage = document.getElementById('viewerStage');
const mainImage = document.getElementById('mainImage');
const overlay = document.getElementById('overlay');

function setStatus(message, isError = false) {
  saveStatus.textContent = message;
  saveStatus.style.color = isError ? '#dc2626' : '#2563eb';
}

function uidForAoi(aoi, index) {
  return `${aoi.image_name || state.currentImage || ''}::${aoi.name || 'unnamed'}::${index}`;
}

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

function getScale() {
  if (!mainImage.naturalWidth || !mainImage.naturalHeight) {
    return { x: 1, y: 1 };
  }
  return {
    x: mainImage.clientWidth / mainImage.naturalWidth,
    y: mainImage.clientHeight / mainImage.naturalHeight,
  };
}

function getOverlayPoint(event) {
  const rect = overlay.getBoundingClientRect();
  return {
    x: clamp(event.clientX - rect.left, 0, rect.width),
    y: clamp(event.clientY - rect.top, 0, rect.height),
  };
}

function displayRectToImageRect(displayRect) {
  const scale = getScale();
  return {
    x: Math.round(displayRect.x / scale.x),
    y: Math.round(displayRect.y / scale.y),
    width: Math.max(1, Math.round(displayRect.width / scale.x)),
    height: Math.max(1, Math.round(displayRect.height / scale.y)),
  };
}

function normalizeDisplayRect(startPoint, endPoint) {
  const x = Math.min(startPoint.x, endPoint.x);
  const y = Math.min(startPoint.y, endPoint.y);
  const width = Math.abs(endPoint.x - startPoint.x);
  const height = Math.abs(endPoint.y - startPoint.y);
  return { x, y, width, height };
}

async function apiGet(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Request failed: ${response.status}`);
  }
  return response.json();
}

async function apiPost(url, payload) {
  const response = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(errorText || `Request failed: ${response.status}`);
  }
  return response.json();
}

function renderImageList() {
  imageListEl.innerHTML = '';

  if (!state.images.length) {
    imageListEl.innerHTML = '<div class="placeholder">No images found.</div>';
    return;
  }

  state.images.forEach((imageName) => {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = `image-item${imageName === state.currentImage ? ' active' : ''}`;
    button.textContent = imageName;
    button.addEventListener('click', () => selectImage(imageName));
    imageListEl.appendChild(button);
  });
}

function renderAoiList() {
  aoiListEl.innerHTML = '';

  if (!state.currentImage || !state.aois.length) {
    aoiListEl.innerHTML = '<div class="placeholder">No AOIs for this image yet.</div>';
    return;
  }

  state.aois.forEach((aoi) => {
    const item = document.createElement('div');
    item.className = `aoi-item${aoi.uid === state.selectedUid ? ' selected' : ''}`;

    const left = document.createElement('div');
    left.style.cursor = 'pointer';
    left.addEventListener('click', () => {
      state.selectedUid = aoi.uid;
      renderAOIs();
      renderAoiList();
    });

    const title = document.createElement('div');
    title.className = 'aoi-item-title';
    title.textContent = aoi.name;

    const meta = document.createElement('div');
    meta.className = 'aoi-item-meta';
    meta.innerHTML = `x: ${aoi.x}, y: ${aoi.y}<br>width: ${aoi.width}, height: ${aoi.height}`;

    left.appendChild(title);
    left.appendChild(meta);

    const deleteBtn = document.createElement('button');
    deleteBtn.type = 'button';
    deleteBtn.className = 'delete-btn';
    deleteBtn.textContent = 'Delete';
    deleteBtn.addEventListener('click', async () => {
      state.aois = state.aois.filter((entry) => entry.uid !== aoi.uid);
      if (state.selectedUid === aoi.uid) {
        state.selectedUid = null;
      }
      renderAOIs();
      renderAoiList();
      await saveCurrentImageAOIs();
    });

    item.appendChild(left);
    item.appendChild(deleteBtn);
    aoiListEl.appendChild(item);
  });
}

function createRectElement(aoi) {
  const scale = getScale();
  const rectEl = document.createElement('div');
  rectEl.className = `aoi-rect${aoi.uid === state.selectedUid ? ' selected' : ''}`;
  rectEl.style.left = `${aoi.x * scale.x}px`;
  rectEl.style.top = `${aoi.y * scale.y}px`;
  rectEl.style.width = `${aoi.width * scale.x}px`;
  rectEl.style.height = `${aoi.height * scale.y}px`;
  rectEl.dataset.uid = aoi.uid;

  const label = document.createElement('div');
  label.className = 'aoi-label';
  label.textContent = aoi.name;
  rectEl.appendChild(label);

  rectEl.addEventListener('mousedown', startDragExistingRect);
  rectEl.addEventListener('click', (event) => {
    event.stopPropagation();
    state.selectedUid = aoi.uid;
    renderAOIs();
    renderAoiList();
  });

  return rectEl;
}

function renderAOIs() {
  overlay.innerHTML = '';
  state.aois.forEach((aoi) => overlay.appendChild(createRectElement(aoi)));

  if (state.tempRect) {
    const temp = document.createElement('div');
    temp.className = 'temp-rect';
    temp.style.left = `${state.tempRect.x}px`;
    temp.style.top = `${state.tempRect.y}px`;
    temp.style.width = `${state.tempRect.width}px`;
    temp.style.height = `${state.tempRect.height}px`;
    overlay.appendChild(temp);
  }
}

function showViewer(hasImage) {
  viewerStage.classList.toggle('hidden', !hasImage);
  emptyState.classList.toggle('hidden', hasImage);
}

async function loadImages() {
  setStatus('Loading images...');
  const data = await apiGet('/api/images');
  state.images = data.images || [];
  if (!state.images.includes(state.currentImage)) {
    state.currentImage = state.images[0] || null;
  }
  renderImageList();

  if (state.currentImage) {
    await selectImage(state.currentImage, false);
  } else {
    showViewer(false);
    state.aois = [];
    renderAoiList();
    currentImageLabel.textContent = 'No image selected';
    setStatus('Idle');
  }
}

async function selectImage(imageName, shouldRenderImageList = true) {
  if (!imageName) {
    return;
  }

  state.currentImage = imageName;
  state.selectedUid = null;
  state.tempRect = null;
  state.aois = [];
  currentImageLabel.textContent = imageName;
  setStatus('Loading AOIs...');
  showViewer(true);

  if (shouldRenderImageList) {
    renderImageList();
  }

  mainImage.src = `/images/${encodeURIComponent(imageName)}?t=${Date.now()}`;
  await loadCurrentImageAOIs();
}

async function loadCurrentImageAOIs() {
  if (!state.currentImage) {
    return;
  }
  const data = await apiGet(`/api/aois?image_name=${encodeURIComponent(state.currentImage)}`);
  state.aois = (data.aois || []).map((aoi, index) => ({ ...aoi, uid: uidForAoi(aoi, index) }));
  renderAOIs();
  renderAoiList();
  setStatus('Loaded');
}

async function saveCurrentImageAOIs() {
  if (!state.currentImage) {
    return;
  }

  clearTimeout(state.saveTimer);
  state.saveTimer = null;
  setStatus('Saving...');

  const payload = {
    image_name: state.currentImage,
    aois: state.aois.map(({ name, x, y, width, height }) => ({
      name,
      x,
      y,
      width,
      height,
    })),
  };

  try {
    await apiPost('/api/aois/save_image', payload);
    setStatus('Saved');
  } catch (error) {
    console.error(error);
    setStatus('Save failed', true);
  }
}

function queueSave(delay = 150) {
  clearTimeout(state.saveTimer);
  state.saveTimer = setTimeout(() => {
    saveCurrentImageAOIs();
  }, delay);
}

function startDraw(event) {
  if (event.target !== overlay || !state.currentImage) {
    return;
  }
  state.isDrawing = true;
  state.drawStart = getOverlayPoint(event);
  state.tempRect = { x: state.drawStart.x, y: state.drawStart.y, width: 0, height: 0 };
  state.selectedUid = null;
  renderAOIs();
  renderAoiList();
}

function startDragExistingRect(event) {
  event.stopPropagation();
  const uid = event.currentTarget.dataset.uid;
  const aoi = state.aois.find((entry) => entry.uid === uid);
  if (!aoi) {
    return;
  }
  state.isDragging = true;
  state.selectedUid = uid;
  state.dragData = {
    uid,
    startPoint: getOverlayPoint(event),
    originalRect: { x: aoi.x, y: aoi.y, width: aoi.width, height: aoi.height },
  };
  renderAOIs();
  renderAoiList();
}

function handlePointerMove(event) {
  if (state.isDrawing && state.drawStart) {
    const endPoint = getOverlayPoint(event);
    state.tempRect = normalizeDisplayRect(state.drawStart, endPoint);
    renderAOIs();
    return;
  }

  if (state.isDragging && state.dragData) {
    const point = getOverlayPoint(event);
    const scale = getScale();
    const dx = Math.round((point.x - state.dragData.startPoint.x) / scale.x);
    const dy = Math.round((point.y - state.dragData.startPoint.y) / scale.y);
    const rect = state.aois.find((entry) => entry.uid === state.dragData.uid);
    if (!rect) {
      return;
    }
    rect.x = clamp(state.dragData.originalRect.x + dx, 0, Math.max(0, mainImage.naturalWidth - rect.width));
    rect.y = clamp(state.dragData.originalRect.y + dy, 0, Math.max(0, mainImage.naturalHeight - rect.height));
    renderAOIs();
    renderAoiList();
    queueSave(120);
  }
}

async function finishPointerAction() {
  if (state.isDrawing && state.tempRect) {
    const imageRect = displayRectToImageRect(state.tempRect);
    state.isDrawing = false;
    state.drawStart = null;
    state.tempRect = null;

    if (imageRect.width > 2 && imageRect.height > 2) {
      let name = aoiNameInput.value.trim();
      if (!name) {
        name = window.prompt('Name for this AOI:')?.trim() || '';
      }

      if (name) {
        const index = state.aois.length;
        state.aois.push({
          uid: uidForAoi({ image_name: state.currentImage, name }, index + Date.now()),
          image_name: state.currentImage,
          name,
          x: imageRect.x,
          y: imageRect.y,
          width: imageRect.width,
          height: imageRect.height,
        });
        aoiNameInput.value = '';
        renderAOIs();
        renderAoiList();
        await saveCurrentImageAOIs();
        return;
      }
    }

    renderAOIs();
    renderAoiList();
    setStatus('Cancelled');
    return;
  }

  if (state.isDragging) {
    state.isDragging = false;
    state.dragData = null;
    await saveCurrentImageAOIs();
  }
}

mainImage.addEventListener('load', () => {
  renderAOIs();
  renderAoiList();
});

overlay.addEventListener('mousedown', startDraw);
document.addEventListener('mousemove', handlePointerMove);
document.addEventListener('mouseup', finishPointerAction);
window.addEventListener('resize', () => renderAOIs());
refreshImagesBtn.addEventListener('click', () => {
  loadImages().catch((error) => {
    console.error(error);
    setStatus('Failed to load images', true);
  });
});

document.addEventListener('DOMContentLoaded', () => {
  loadImages().catch((error) => {
    console.error(error);
    setStatus('Failed to start', true);
  });
});
