import { initializeApp } from "https://www.gstatic.com/firebasejs/10.12.2/firebase-app.js";
import {
  getAuth,
  createUserWithEmailAndPassword,
  signInWithEmailAndPassword
} from "https://www.gstatic.com/firebasejs/10.12.2/firebase-auth.js";

const firebaseConfig = window.FIREBASE_WEB_CONFIG || null;

function showError(message) {
  const box = document.getElementById("authError");
  if (!box) return;
  // Map Firebase error codes to friendly messages
  const friendly = {
    "auth/user-not-found":       "No account found with this email. Please register first.",
    "auth/wrong-password":       "Incorrect password. Please try again.",
    "auth/invalid-credential":   "Invalid email or password. Please check and try again.",
    "auth/email-already-in-use": "An account with this email already exists. Please login instead.",
    "auth/weak-password":        "Password must be at least 6 characters.",
    "auth/invalid-email":        "Please enter a valid email address.",
    "auth/too-many-requests":    "Too many failed attempts. Please wait a moment and try again.",
    "auth/network-request-failed": "Network error. Please check your internet connection.",
  };
  const code = message.match(/\(([^)]+)\)/)?.[1] || "";
  box.textContent = friendly[code] || message;
  box.classList.remove("d-none");
  document.getElementById("authSuccess")?.classList.add("d-none");
}

function showSuccess() {
  document.getElementById("authError")?.classList.add("d-none");
  document.getElementById("authSuccess")?.classList.remove("d-none");
}

function setLoading(btnId, spinId, textId, loading) {
  const btn  = document.getElementById(btnId);
  const spin = document.getElementById(spinId);
  const text = document.getElementById(textId);
  if (!btn) return;
  btn.disabled = loading;
  spin?.classList.toggle("d-none", !loading);
  text?.classList.toggle("d-none", loading);
}

async function createServerSession(idToken) {
  const response = await fetch("/session_login", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ idToken })
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok || !data.success) {
    throw new Error(data.message || "Unable to create session.");
  }
  return data;
}

if (!firebaseConfig) {
  document.getElementById("authError")?.classList.remove("d-none");
  if (document.getElementById("authError"))
    document.getElementById("authError").textContent = "Firebase configuration is missing.";
} else {
  const app  = initializeApp(firebaseConfig);
  const auth = getAuth(app);

  // ── LOGIN ──
  const loginForm = document.getElementById("loginForm");
  if (loginForm) {
    loginForm.addEventListener("submit", async (e) => {
      e.preventDefault();
      document.getElementById("authError")?.classList.add("d-none");
      setLoading("loginSubmit", "loginSpinner", "loginBtnText", true);

      try {
        const email    = document.getElementById("loginEmail").value.trim();
        const password = document.getElementById("loginPassword").value;

        const cred    = await signInWithEmailAndPassword(auth, email, password);
        const idToken = await cred.user.getIdToken();
        showSuccess();
        const session = await createServerSession(idToken);
        window.location.assign(session.redirect || "/dashboard");
      } catch (err) {
        showError(err.message || "Login failed.");
        setLoading("loginSubmit", "loginSpinner", "loginBtnText", false);
      }
    });
  }

  // ── REGISTER ──
  const registerForm = document.getElementById("registerForm");
  if (registerForm) {
    registerForm.addEventListener("submit", async (e) => {
      e.preventDefault();
      document.getElementById("authError")?.classList.add("d-none");
      setLoading("registerSubmit", "registerSpinner", "registerBtnText", true);

      try {
        const email    = document.getElementById("registerEmail").value.trim();
        const password = document.getElementById("registerPassword").value;

        const cred    = await createUserWithEmailAndPassword(auth, email, password);
        const idToken = await cred.user.getIdToken();
        showSuccess();
        const session = await createServerSession(idToken);
        window.location.assign(session.redirect || "/dashboard");
      } catch (err) {
        showError(err.message || "Registration failed.");
        setLoading("registerSubmit", "registerSpinner", "registerBtnText", false);
      }
    });
  }
}
