%include('layout.tpl')

<div class="container-fluid">
 <div class="row">
  <div class="col-8">
   <div class="content-container">
    <h4 class="lcm-heading">Land Cover Mapping</h4>
    <p>Human centric land cover mapping.</p>
    <p>Interactively train machine learning models on fit for purpose imagery in order to produce downloadable land cover maps.</p>
    <form action="/login" method="GET">
    <p><button type="submit" class="submit-button" id="submit" onClick="ShowLoading()">LOG IN</button></p>
    </form>
   </div>
  </div>
 </div>
</div>
</body>

</html>