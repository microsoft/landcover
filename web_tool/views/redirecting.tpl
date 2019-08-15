 %include('layout.tpl')

<div class="container">
	<div class="row">
		<div class="content-container not-authorized-container">
     <h5>Checking access, Please wait...</h5><br/>
		</div>
	</div>
</div>
<form action="/checkAccess" method="post">
 <input id="token" name="token" type="hidden" value=""/>
</form>
</body>
</html>
<script>
	CheckAccessToken()
</script>