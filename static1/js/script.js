document.addEventListener("DOMContentLoaded", function () {
  const form = document.getElementById("query-form");
  const thinking = document.getElementById("thinking");

  form.addEventListener("submit", function () {
    thinking.style.display = "block";
  });
});
