const API_BASE_URL = "https://avatar-722b.onrender.com";

const factors = [
  ["smoking", "Smoking", 0.0],
  ["uv_exposure", "UV exposure", 0.4],
  ["alcohol", "Alcohol", 0.1],
  ["stress", "Stress", 0.6],
  ["diet", "Unhealthy diet", 0.3],
];

const form = document.querySelector("#avatar-form");
const factorControls = document.querySelector("#factor-controls");
const statusElement = document.querySelector("#status");
const resultsElement = document.querySelector("#results");
const originalPreview = document.querySelector("#original-preview");
const agedResult = document.querySelector("#aged-result");
const featureList = document.querySelector("#feature-list");
const generateButton = document.querySelector("#generate-button");

for (const [key, label, initialValue] of factors) {
  const wrapper = document.createElement("div");
  wrapper.className = "factor-row";

  const title = document.createElement("span");
  title.textContent = label;

  const value = document.createElement("output");
  value.textContent = initialValue.toFixed(1);

  const slider = document.createElement("input");
  slider.type = "range";
  slider.min = "0";
  slider.max = "1";
  slider.step = "0.1";
  slider.value = String(initialValue);
  slider.dataset.factor = key;

  slider.addEventListener("input", () => {
    value.textContent = Number(slider.value).toFixed(1);
  });

  wrapper.append(title, value, slider);
  factorControls.append(wrapper);
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const imageInput = document.querySelector("#image");
  const image = imageInput.files[0];

  if (!image) {
    statusElement.textContent = "Please select a portrait.";
    return;
  }

  const factorValues = {};

  for (const slider of document.querySelectorAll("[data-factor]")) {
    factorValues[slider.dataset.factor] = Number(slider.value);
  }

  const profile = {
    age: Number(document.querySelector("#age").value),
    target_age: Number(document.querySelector("#target-age").value),
    sex: document.querySelector("#sex").value,
    ethnicity: document.querySelector("#ethnicity").value.trim(),
    factors: factorValues,
  };

  const formData = new FormData();
  formData.append("image", image);
  formData.append("profile_json", JSON.stringify(profile));

  originalPreview.src = URL.createObjectURL(image);
  resultsElement.classList.add("hidden");
  generateButton.disabled = true;
  statusElement.textContent =
    "Generating portrait. The first request may take longer while the server wakes up.";

  try {
    const response = await fetch(`${API_BASE_URL}/api/generate`, {
      method: "POST",
      body: formData,
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.detail || "Generation failed.");
    }

    agedResult.src =
      `data:${data.image.mime_type};base64,${data.image.base64}`;

    featureList.replaceChildren();

    const features = Array.isArray(data.features)
      ? data.features.slice(0, 8)
      : [];

    for (const feature of features) {
      const item = document.createElement("li");

      if (typeof feature === "string") {
        item.textContent = feature;
      } else {
        const name =
          feature.label ||
          feature.name ||
          feature.feature ||
          feature.node_id ||
          "Aging feature";

        const score =
          feature.score ??
          feature.value ??
          feature.activation;

        item.textContent =
          score === undefined
            ? name
            : `${name}: ${Number(score).toFixed(2)}`;
      }

      featureList.append(item);
    }

    resultsElement.classList.remove("hidden");
    statusElement.textContent = "Generation complete.";
  } catch (error) {
    statusElement.textContent = error.message;
  } finally {
    generateButton.disabled = false;
  }
});