document.addEventListener("DOMContentLoaded", () => {
  const copyButton = document.getElementById("copy-resume");
  const resumePreview = document.getElementById("resume-preview");
  const copyStatus = document.getElementById("copy-status");
  if (copyButton && resumePreview) {
    copyButton.addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(resumePreview.innerText);
        copyStatus.textContent = "Draft copied. Review it before use.";
      } catch (_error) {
        copyStatus.textContent = "Copy is unavailable here. Select the draft text and copy it manually.";
      }
    });
  }

  const form = document.querySelector(".analysis-form");
  const input = document.getElementById("resume");
  const dropZone = document.getElementById("upload-box");
  const uploadTitle = document.getElementById("upload-title");
  const uploadSubtitle = document.getElementById("upload-subtitle");
  const uploadError = document.getElementById("upload-error");
  const description = document.getElementById("job_description");
  const descriptionCount = document.getElementById("description-count");
  const submitButton = document.getElementById("submit-button");
  if (!form || !input || !dropZone) return;

  const maxBytes = 8 * 1024 * 1024;
  const allowed = new Set(["pdf", "docx", "txt"]);

  function showUploadError(message) {
    uploadError.textContent = message;
    uploadError.hidden = !message;
    dropZone.classList.toggle("invalid", Boolean(message));
  }

  function displayFile(file) {
    if (!file) return;
    const extension = file.name.split(".").pop().toLowerCase();
    if (!allowed.has(extension)) {
      input.value = "";
      showUploadError("Choose a PDF, DOCX, or TXT resume.");
      return;
    }
    if (file.size > maxBytes) {
      input.value = "";
      showUploadError("That file is over 8 MB. Choose a smaller resume.");
      return;
    }
    showUploadError("");
    dropZone.classList.add("has-file");
    uploadTitle.textContent = file.name;
    uploadSubtitle.textContent = `${(file.size / 1024).toFixed(0)} KB · ready to analyze`;
  }

  input.addEventListener("change", () => displayFile(input.files[0]));
  ["dragenter", "dragover"].forEach((eventName) => {
    dropZone.addEventListener(eventName, (event) => {
      event.preventDefault();
      dropZone.classList.add("dragging");
    });
  });
  ["dragleave", "drop"].forEach((eventName) => {
    dropZone.addEventListener(eventName, (event) => {
      event.preventDefault();
      dropZone.classList.remove("dragging");
    });
  });
  dropZone.addEventListener("drop", (event) => {
    const file = event.dataTransfer.files[0];
    if (!file) return;
    const transfer = new DataTransfer();
    transfer.items.add(file);
    input.files = transfer.files;
    displayFile(file);
  });

  function updateCount() {
    const length = description.value.trim().length;
    descriptionCount.textContent = `${length} / 80`;
    descriptionCount.classList.toggle("complete", length >= 80);
  }
  description.addEventListener("input", updateCount);
  updateCount();

  form.addEventListener("submit", (event) => {
    if (!form.reportValidity()) {
      event.preventDefault();
      return;
    }
    submitButton.disabled = true;
    submitButton.setAttribute("aria-busy", "true");
    submitButton.innerHTML = '<span class="button-spinner" aria-hidden="true"></span> Building your brief…';
  });
});
