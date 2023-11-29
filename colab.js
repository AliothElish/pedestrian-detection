function ClickConnect() {
  console.log('Connect pushed')
  document
    .querySelector('#top-toolbar > colab-connect-button')
    .shadowRoot.querySelector('#connect')
    .click()
}

var id = setInterval(ClickConnect, 60000)
