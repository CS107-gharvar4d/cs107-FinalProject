
## Introducción

ADG4 es un paquete accessible y transparente que calcula un valor usando diferenciacion y calcular derivados de ese valor. Nuestro esquema 'Open Source' busca ser lo mas incluyente y cualquier persona de cualquier genero o nacionalidad pueden aportar al proyecto. 

ADG4 integra un conjunto de técnicas para evaluar la derivada de varias expresiones matemáticas. Ser capaz de tomar muchas derivadas tiene importantes aplicaciones prácticas en una variedad de dominios. En física, la derivada de primer orden nos dice la tasa de cambio, mientras que otras órdenes nos pueden decir más piezas de información como aceleración y actergia. En estadística, las aplicaciones incluyen bayesianas la inferencia y el entrenamiento de las redes neuronales. En economía, tomando la derivada de ganancia y utilidad La función permite a los agentes maximizar los resultados esperados. Estos son solo algunos casos en los que diferenciar funciones complejas es útil y necesario.

Dado el amplio conjunto de aplicaciones en todos los dominios, esperamos que pueda ser una herramienta útil ADG4 que puede facilitar la evaluación numérica rápida y sencilla de derivadas a través de la computación.

## Cómo utilizar

Estos son los pasos para instalar y utilizar el modulo ADG4. Primero, a continuación se muestran los pasos para descargar e instalar `ADG4`;

1. Crea y activa un entorno virtual, ya sea con condón

```
conda create -n adg4_env python=3.8
conda activate adg4_env 
```

o cualquier método preferido
 
2. Descarga nuestro repositorio, `git clone git@github.com: CS107-gharvar4d / cs107-FinalProject.git`
3. Navegue hasta la carpeta con `CD-cs107 FinalProject`
4. Instale los requisitos con `pip install -r requirements.txt`
5. Instale nuestro paquete con `pip install --editable. / Code`
6. Ahora puede hacer `importar ADG4.ad como anuncio` o simplemente ejecutar nuestras pruebas con los comandos` pytest` en el directorio del repositorio


## Cómo contribuir al proyecto

Es importante considerar los posibles efectos en la creación de software y ponerlo a disposición de todos sin distinción. Nuestra filosofía es distribuir software a cualquier persona para cualquier propósito y hacer esfuerzos para desarrollarlo en un asunto público colaborativo. Un buen comienzo para esto es el hecho de que todo nuestro código fuente reside en Github, una plataforma abierta donde todos pueden colaborar. Incluya documentación para personas que no hablen inglés, como español, y habrá más en el futuro.

Nuestro grupo acepta dos flujos de trabajo comunes para la colaboración:

1. Shared repo

Clone nuestro repositorio y actualice con `git pull origin master`, luego cree una rama de trabajo con` git checkout -b MyNewBranch` y realice cualquier cambio antes de la puesta en escena.
Confirme localmente y cargue los cambios (incluida su nueva rama) en GitHub con `git push origin MyNewBranch`

Luego, navega a main en GitHub donde ahora deberías ver tu nueva rama. Haga clic en el botón "Solicitud de extracción" y "Enviar solicitud de extracción"

2. Fork and Pull

Podemos ceder derechos a “Colaboradores”. Aunque los colaboradores no tienen acceso push al upstream, aceptamos Pull Requests (PR) de ellos, revisamos y luego fusionamos los cambios en el repositorio principal si se aprueban.

Todos los colaboradores de ADG4 deben adherirse a los estándares de NeurIPS, permanecer atentos y evaluar el impacto de su código por comportamiento poco ético o uso ilegal. Algunas aplicaciones de software en las que esto puede suceder son la seguridad o la privacidad. Si se conoce algún uso indebido, se debe contactar al líder del proyecto de inmediato.

