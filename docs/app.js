const API_BASE_URL = "https://avatar-722b.onrender.com";

const GOOGLE_FORM_BASE_URL =
  "https://docs.google.com/forms/d/e/1FAIpQLSfd0SfEgYtIe8uOUszcq6-Cu_qaUArEBAoa42Warz-BnGeBOA/viewform";

const GOOGLE_FORM_FIELDS = {
  currentAge: "entry.1014704484",
  gender: "entry.1166027910",
  ethnicity: "entry.739325734",
  smoking: "entry.1869532955",
  uvExposure: "entry.950627133",
  alcohol: "entry.158385039",
  stress: "entry.1193685801",
  diet: "entry.265219643",
};

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
// const downloadButton = document.querySelector("#download-button");
const consentModal = document.querySelector("#consent-modal");
const consentRead = document.querySelector("#consent-read");
const consentImage = document.querySelector("#consent-image");
const consentAge = document.querySelector("#consent-age");
const acceptConsent = document.querySelector("#accept-consent");
const declineConsent = document.querySelector("#decline-consent");
const consentMessage = document.querySelector("#consent-message");
const feedbackButton =
  document.querySelector("#feedback-button");

let latestFeedbackData = null;

// let latestGeneratedImage = null;

function hasAcceptedConsent() {
  return sessionStorage.getItem("avatar-pilot-consent") === "accepted";
}

function updateConsentButton() {
  const allChecked =
    consentRead.checked &&
    consentImage.checked &&
    consentAge.checked;

  acceptConsent.disabled = !allChecked; 
}

function openConsentModal() {
  consentModal.classList.remove("hidden");
  document.body.classList.add("modal-open");
}

function closeConsentModal() {
  consentModal.classList.add("hidden");
  document.body.classList.remove("modal-open");
}

consentRead.addEventListener("change", updateConsentButton);
consentImage.addEventListener("change", updateConsentButton);
consentAge.addEventListener("change", updateConsentButton);

acceptConsent.addEventListener("click", () => {
  if (
    !consentRead.checked ||
    !consentImage.checked ||
    !consentAge.checked
  ) {
    consentMessage.textContent =
      "Please confirm all three statements before continuing.";
    return;
  }

  sessionStorage.setItem("avatar-pilot-consent", "accepted");
  consentMessage.textContent = "";
  closeConsentModal();
});

declineConsent.addEventListener("click", () => {
  sessionStorage.removeItem("avatar-pilot-consent");

  const consentCard = document.querySelector(".consent-card");

  consentCard.innerHTML = `
    <h2>Participation declined</h2>

    <p>
      You have chosen not to participate in the AVATAR pilot study.
      No image has been uploaded or processed.
    </p>

    <p>
      You may close this page.
    </p>
  `;
});

// if (hasAcceptedConsent()) {
//   closeConsentModal();
// } else {
//   openConsentModal();
// }

// Temporarily disabled before IRB approval
closeConsentModal();

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
    sex: document.querySelector("#sex").value.trim(),
    ethnicity: document.querySelector("#ethnicity").value.trim(),
    factors: factorValues,
  };

  const formData = new FormData();
  formData.append("image", image);
  formData.append("profile_json", JSON.stringify(profile));

  originalPreview.src = URL.createObjectURL(image);
  resultsElement.classList.add("hidden");
  generateButton.disabled = true;
  // downloadButton.disabled = true;
  // latestGeneratedImage = null;
  feedbackButton.disabled = true;
  latestFeedbackData = null;

  featureList.innerHTML =
  '<li class="feature-empty">Analyzing aging features...</li>';
  statusElement.textContent =
    "Generating portrait. The first request may take longer while the server wakes up.";

  try {
    const response = await fetch(`${API_BASE_URL}/api/generate`, {
      method: "POST",
      body: formData,
    });

    const data = await response.json();
    console.log("Response keys:", Object.keys(data));
    console.log("Knowledge keys:", Object.keys(data.knowledge || {}));

    if (!response.ok) {
      throw new Error(data.detail || "Generation failed.");
    }

    agedResult.src =
      `data:${data.image.mime_type};base64,${data.image.base64}`;
    
    latestFeedbackData = {
      currentAge: profile.age,
      gender: profile.sex,
      ethnicity: profile.ethnicity,

      smoking: Math.round(profile.factors.smoking * 10),
      uvExposure: Math.round(profile.factors.uv_exposure * 10),
      alcohol: Math.round(profile.factors.alcohol * 10),
      stress: Math.round(profile.factors.stress * 10),
      diet: Math.round(profile.factors.diet * 10),
    };

feedbackButton.disabled = false;


    // latestGeneratedImage = {
    //   mimeType: data.image.mime_type,
    //   base64: data.image.base64,
    // };

// downloadButton.disabled = false;

    featureList.replaceChildren();

    const formatFeatureName = (name) =>
      name
        .replaceAll("_", " ")
        .replace(/\b\w/g, (letter) => letter.toUpperCase());

    const representation =
      data.aging_representation ||
      data.knowledge?.aging_representation ||
      data.result?.aging_representation ||
      data.output?.aging_representation ||
      {};

    console.log("Full API response:", data);
    console.log("Resolved aging representation:", representation);

    const features = Object.entries(representation)
      .map(([name, score]) => ({
        name,
        score: Number(score),
      }))
      .filter(({ score }) => Number.isFinite(score) && score > 0)
      .sort((a, b) => b.score - a.score)
      .slice(0, 8);

    for (const feature of features) {
      const item = document.createElement("li");
      item.className = "feature-item";

      const header = document.createElement("div");
      header.className = "feature-header";

      const nameElement = document.createElement("span");
      nameElement.className = "feature-name";
      nameElement.textContent = formatFeatureName(feature.name);

      const scoreElement = document.createElement("span");
      scoreElement.className = "feature-score";
      scoreElement.textContent = feature.score.toFixed(2);

      const track = document.createElement("div");
      track.className = "feature-track";

      const bar = document.createElement("div");
      bar.className = "feature-bar";

      const normalizedScore = Math.max(0, Math.min(1, feature.score));
      bar.style.width = `${normalizedScore * 100}%`;

      header.append(nameElement, scoreElement);
      track.append(bar);
      item.append(header, track);
      featureList.append(item);
    }

    if (features.length === 0) {
      const emptyItem = document.createElement("li");
      emptyItem.className = "feature-empty";
      emptyItem.textContent =
        "No aging features were found in the API response.";
      featureList.append(emptyItem);
    }


    resultsElement.classList.remove("hidden");
    statusElement.textContent = "Generation complete.";
  } catch (error) {
    statusElement.textContent = error.message;
  } finally {
    generateButton.disabled = false;
  }
});

// downloadButton.addEventListener("click", () => {
//   if (!latestGeneratedImage?.base64) {
//     statusElement.textContent =
//       "No generated image is available to download.";
//     return;
//   }

//   try {
//     const binaryString = atob(latestGeneratedImage.base64);
//     const bytes = new Uint8Array(binaryString.length);

//     for (let index = 0; index < binaryString.length; index += 1) {
//       bytes[index] = binaryString.charCodeAt(index);
//     }

//     const mimeType =
//       latestGeneratedImage.mimeType || "image/png";

//     const imageBlob = new Blob(
//       [bytes],
//       { type: mimeType }
//     );

//     const objectUrl = URL.createObjectURL(imageBlob);

//     const extension =
//       mimeType === "image/jpeg"
//         ? "jpg"
//         : mimeType === "image/webp"
//           ? "webp"
//           : "png";

//     const now = new Date();

//     const timestamp = [
//       now.getFullYear(),
//       String(now.getMonth() + 1).padStart(2, "0"),
//       String(now.getDate()).padStart(2, "0"),
//       "_",
//       String(now.getHours()).padStart(2, "0"),
//       String(now.getMinutes()).padStart(2, "0"),
//       String(now.getSeconds()).padStart(2, "0"),
//     ].join("");

//     const downloadLink = document.createElement("a");

//     downloadLink.href = objectUrl;
//     downloadLink.download =
//       `AVATAR_aged_portrait_${timestamp}.${extension}`;

//     document.body.appendChild(downloadLink);
//     downloadLink.click();
//     downloadLink.remove();

//     setTimeout(() => {
//       URL.revokeObjectURL(objectUrl);
//     }, 1000);

//     statusElement.textContent =
//       "Aged portrait downloaded successfully.";
//   } catch (error) {
//     console.error("Download failed:", error);

//     statusElement.textContent =
//       "The portrait could not be downloaded. Please try again.";
//   }
// });

feedbackButton.addEventListener("click", () => {
  if (!latestFeedbackData) {
    statusElement.textContent =
      "Please generate an aged portrait before continuing.";
    return;
  }

  const params = new URLSearchParams();

  params.set("usp", "pp_url");

  params.set(
    GOOGLE_FORM_FIELDS.currentAge,
    String(latestFeedbackData.currentAge)
  );

  params.set(
    GOOGLE_FORM_FIELDS.gender,
    latestFeedbackData.gender
  );

  params.set(
    GOOGLE_FORM_FIELDS.ethnicity,
    latestFeedbackData.ethnicity
  );

  params.set(
    GOOGLE_FORM_FIELDS.smoking,
    String(latestFeedbackData.smoking)
  );

  params.set(
    GOOGLE_FORM_FIELDS.uvExposure,
    String(latestFeedbackData.uvExposure)
  );

  params.set(
    GOOGLE_FORM_FIELDS.alcohol,
    String(latestFeedbackData.alcohol)
  );

  params.set(
    GOOGLE_FORM_FIELDS.stress,
    String(latestFeedbackData.stress)
  );

  params.set(
    GOOGLE_FORM_FIELDS.diet,
    String(latestFeedbackData.diet)
  );

  const feedbackUrl =
    `${GOOGLE_FORM_BASE_URL}?${params.toString()}`;

  window.open(feedbackUrl, "_blank", "noopener");
});